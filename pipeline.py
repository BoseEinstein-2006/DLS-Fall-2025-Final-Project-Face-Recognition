# импортируем все что нужно

import torch
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import math
from torchvision import models
import pandas as pd
import cv2

############################################
#         StackedHourglass модель          #
############################################

###### Взято из 2_Face_aligment.ipynb ######

# реализация взята прямиком из ноутбука с заданием

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)

# Ну собственно сам Hourglass модуль
# Сначала уменьшает разрешение (encoder), затем увеличивает (decoder), при этом используя skip-connections для сохранения деталей

class Hourglass(nn.Module):

    def __init__(self, depth, channels):
        super().__init__()
        self.depth = depth  # глубина рекурсии (сколько раз будем уменьшать разрешение)
        self.channels = channels  # количество каналов во всех слоях
        
        # Верхняя ветка (skip connection) - сохраняет исходное разрешение
        # Это параллельный путь, который идёт в обход downsample/upsample
        self.res_up = ResidualBlock(channels, channels)
        
        # Нижняя ветка - путь с уменьшением/увеличением разрешения:
        self.res_low1 = ResidualBlock(channels, channels)  # обработка перед downsample
        self.down = nn.MaxPool2d(2, 2)  # уменьшаем разрешение в 2 раза (H/2, W/2)
        self.res_low2 = ResidualBlock(channels, channels)  # обработка после downsample
        
        # Рекурсивная часть:
        if depth > 1:
            # Если ещё не достигли дна, создаём вложенный hourglass меньшей глубины
            self.low = Hourglass(depth - 1, channels)
        else:
            # Если depth == 1, это самый нижний уровень (bottleneck)
            # Просто обрабатываем признаки без дальнейшего downsample
            self.low = ResidualBlock(channels, channels)
        
        # Обработка после рекурсивной части, перед upsample
        self.res_low3 = ResidualBlock(channels, channels)
        
    def forward(self, x):
        # Верхняя ветка (skip connection)
        # Обрабатываем входные данные на исходном разрешении
        up1 = self.res_up(x)
        
        # Нижняя ветка (encoder-decoder)
        # Подготовка к понижению разрешения
        low = self.res_low1(x)
        
        # Уменьшаем разрешение в 2 раза 
        low = self.down(low)
        
        # Обрабатываем признаки на пониженном разрешении
        low = self.res_low2(low)
        
        # Рекурсивная часть или bottleneck:
        # Если depth > 1, вызываем вложенный Hourglass
        # Если depth == 1, просто обрабатываем через ResidualBlock
        low = self.low(low)
        
        # Ещё одна обработка после рекурсивной части
        low = self.res_low3(low)
        
        # Восстанавливаем разрешение обратно 
        up2 = F.interpolate(low, scale_factor=2, mode="nearest")
        
        # Складываем результаты верхней и нижней веток
        return up1 + up2

# голова сети, где мы преобразует признаки в тепловые карты для каждого landmark

class Head(nn.Module):
    def __init__(self, channels, num_landmarks):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), 
            nn.ReLU(inplace=True), 
            # Каждый канал выхода = хитмап для одного landmark
            nn.Conv2d(channels, num_landmarks, 1)
        )
        
    def forward(self, x):
        # На входе (B, channels, H, W)
        # На выходе (B, num_landmarks, H, W)
        return self.block(x)
    
# теперь стэкаем несколько Hourglass модулей
# каждый Hourglass предсказывает хитмапы и эти предсказания используютсядля улучшения следующего Hourglass в стеке

class StackedHourglass(nn.Module):
    def __init__(self, num_stacks, num_landmarks, channels, depth):
        super().__init__()
        self.num_stacks = num_stacks  # количество последовательных Hourglass модулей
        self.num_landmarks = num_landmarks  # количество landmark для детекции
        
        # начальный блок обработки картинки на входе
        # берем сырые пиксели, сжимает картинку, делает больше каналов и вытаскивает базовые штуки вроде краёв и форм
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, channels),
        )
        
        # создаём список из num_stacks Hourglass модулей
        # каждый Hourglass имеет одинаковую архитектуру
        self.hourglasses = nn.ModuleList([
            Hourglass(depth, channels) for _ in range(num_stacks)
        ])
        
        # создаём список голов, по одной на каждый Hourglass
        # каждая голова генерирует хитмапы
        self.heads = nn.ModuleList([
            Head(channels, num_landmarks) for _ in range(num_stacks)
        ])
        
        # Связб между Hourglass
        self.feat_to_feat = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in range(num_stacks - 1)
        ])
        
        # Преобразование предсказанных хитмапов обратно в пространство признаков
        self.hm_to_feat = nn.ModuleList([
            nn.Conv2d(num_landmarks, channels, 1) for _ in range(num_stacks - 1)
        ])
        
    def forward(self, x):
        # пропускаем изображение через stem
        # На входе(B, 3, H, W) 
        # На выходе (B, channels, H/4, W/4)
        x = self.stem(x)
        
        # Список для сохранения всех промежуточных предсказаний
        outputs = []
        
        # Проходим через каждый Hourglass в стеке
        for i in range(self.num_stacks):
            # Пропускаем признаки через i-ый Hourglass
            feat = self.hourglasses[i](x) 
            
            # Генерируем хитмапы из признаков
            hm = self.heads[i](feat)  
            
            # Сохраняем предсказания для loss на каждом уровне
            outputs.append(hm)
            
            # Если это не последний стек, то подготавливаем вход для следующего
            if i < self.num_stacks - 1:
                # Складываем исходные признаки x, новые признаки из текущего Hourglass feat и уже предсказанные хитмапы hm
                x = x + self.feat_to_feat[i](feat) + self.hm_to_feat[i](hm)
        
        # Возвращаем список всех предсказаний
        return outputs
    
# переводим хитмары в точки
@torch.no_grad()
def heatmaps_to_points(hm):
    B, N, H, W = hm.shape 
    flat = hm.view(B, N, -1) # разворачиваем H*W в одно измерение
    idx = flat.argmax(dim=-1) # находим индекс максимального значения         
    
    # преобразуем линейный индекс обратно в 2D координаты
    y = idx // W
    x = idx % W
    
    # объединяем x и y в один тензор формата (B, N, 2)
    pts = torch.stack([x, y], dim=-1).float()
    return pts

# преобразует координаты из системы координат хитмапа в систему координат исходного изображения
@torch.no_grad()
def hm_points_to_img_points(pts_hm, input_size, hm_size):
    # вычисляем коэффициент масштабирования между хитмапом и изображением
    scale = input_size / float(hm_size)
    return pts_hm * scale

# эта функция задает координаты ключевый точек на выровненной кратинки
# тоесть мы тут руками задаем где должен находится левый глаз, где правый и т.д.
# я чутка поигрался с этими цифрами, вроде они ок сейчас

def get_canonical_5pts(output_size):
    W = H = output_size
    return np.array([
        [0.30 * W, 0.35 * H],  # левый глаз
        [0.70 * W, 0.35 * H],  # правый глаз
        [0.50 * W, 0.52 * H],  # нос
        [0.35 * W, 0.75 * H],  # левый угол рта
        [0.65 * W, 0.75 * H],  # правый угол рта
    ], dtype=np.float32)

# тут мы выравниваем картинку с помощью аффинного преобразования
# тоесть берем на вход предсказание модели src, говорим как должны быть расположены ключевые точки dst, находим нужное преобразование M и с его помощью меняет исходную картинку
def align_face_affine(img_rgb, src_pts_5, output_size):

    src = np.asarray(src_pts_5, dtype=np.float32)
    dst = get_canonical_5pts(output_size=output_size)

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    aligned = cv2.warpAffine(img_rgb, M, (output_size, output_size), flags=cv2.INTER_LINEAR)
    return aligned


############################################


###### Написано дополнительно для pipeline.py ######

# предсказание ключевых точек для батча изображений
@torch.no_grad()
def predict_points_batch(model, images, device, hm_size, input_size):
    outputs = model(images.to(device).float())
    pred_hm = outputs[-1]
    pred_pts_hm = heatmaps_to_points(pred_hm)
    pred_pts_img = hm_points_to_img_points(pred_pts_hm, input_size=input_size, hm_size=hm_size)
    return pred_pts_img.cpu().numpy()

# загружаем чекпоинт Hourglass-модели 
def load_hourglass_checkpoint(hg_path, device="cpu"):
    ckpt = torch.load(hg_path, map_location=device)

    meta = ckpt.get("meta", {})

    # строим модель на основе мета-информации
    model = StackedHourglass(
        num_stacks=int(meta["num_stacks"]),
        num_landmarks=int(meta["num_landmarks"]),
        channels=int(meta.get("channels", 256)),
        depth=int(meta.get("depth", 4)),
    ).to(device)

    # загружаем веса
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt

__all__ = ['ResidualBlock','Hourglass','Head','StackedHourglass','heatmaps_to_points','hm_points_to_img_points','get_canonical_5pts','align_face_affine','predict_points_batch', 'load_hourglass_checkpoint']


############################################
#         Arcface модель          #
############################################


###### Взято из 3_Face_recognition.ipynb ######

# ArcFace голова
class ArcMarginProduct(nn.Module):
 
    def __init__(self, in_features, out_features, s, m):
        super().__init__()

        self.in_features = in_features # размерность эмбеддинга
        self.out_features = out_features # число identites

        self.s = float(s) # масштаб логитов 
        self.m = float(m) # угловой маржин

        self.weight = nn.Parameter(torch.empty(out_features, in_features)) # матрица весов классов, каждая строка это "центр" класса в пространстве эмбеддингов
        nn.init.xavier_uniform_(self.weight) 

        # Предварительно считаем cos(m), sin(m) для формулы:
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        self.cos_m = math.cos(self.m) 
        self.sin_m = math.sin(self.m)

        # порог и поправка для численной стабильности 
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embeddings, labels=None):

        # L2-нормализация эмбеддингов и весов:
        embeddings = F.normalize(embeddings, p=2, dim=1)      
        weight = F.normalize(self.weight, p=2, dim=1)        

        # перемножаем их
        cosine = F.linear(embeddings, weight)              

        # Если labels нет, то возвращаем чистые cosine * s
        # маржин нужен только при обучении чтобы формировать пространство эмбеддингов
        if labels is None:
            return cosine * self.s

        # считаем sin(theta) = sqrt(1 - cos^2(theta))
        # clamp нужен, чтобы не получить отрицательное из-за численных ошибок
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=0.0))

        # читаем phi = cos(theta + m) по формуле, вот и пригодилась тригонометрия!
        phi = cosine * self.cos_m - sine * self.sin_m        

        # если cosine слишком маленький (угол тета слишком большой), то вместе phi = cos(theta + m) используем phi = cosine - mm
        # это делается для численной стабильности, потому что если teta + m окажется больше pi, то cos перестанет быть монотонной функцией угла и у нас будут проблемы с обучением
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # оne-hot для выбора ground-truth в каждом примере
        one_hot = torch.zeros_like(cosine)                    
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # для правильного класса phi = cos(theta + m)
        # для остальных классов cosine = cos(theta)
        logits = one_hot * phi + (1.0 - one_hot) * cosine

        # масштабируем логиты
        logits = logits * self.s
        return logits

# сама ArcFace модель

class ArcFaceModel(nn.Module):
    def __init__(self, backbone, embedder, head):
        super().__init__()
        self.backbone = backbone
        self.embedder = embedder
        self.head = head

    def forward(self, x, labels=None):
        # backbone извлекает признаки
        features = self.backbone(x)

        # embedder переводит признаки в пространстве эмбеддингов
        emb = self.embedder(features)

        # ArcFace голова считает логиты
        logits = self.head(emb, labels)

        return logits, emb

@torch.no_grad()
def arcface_forward_embeddings(model, x):
    feat = model.backbone(x)
    z = model.embedder(feat)
    z = z.float()
    z = F.normalize(z, p=2, dim=1)
    return z

###### Написано дополнительно для pipeline.py ######

# строим ArcFace-модель с ResNet50-бэкбоном
def build_arcface_model(num_classes, emb_dim, s, m, device):

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Identity()  

    embedder = nn.Sequential(
        nn.Linear(in_features, emb_dim, bias=False),
        nn.BatchNorm1d(emb_dim),
    )

    head = ArcMarginProduct(emb_dim, num_classes, s=s, m=m)
    model = ArcFaceModel(backbone, embedder, head).to(device)
    return model

# загружаем чекпоинта ArcFace-модели
def load_arcface_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    num_classes = len(ckpt["classes"])

    emb_dim = int(ckpt.get("emb_dim", 512))
    s = float(ckpt.get("arcface_s", 30.0))
    m = float(ckpt.get("arcface_m", 0.50))

    model = build_arcface_model(num_classes=num_classes, emb_dim=emb_dim, s=s, m=m, device=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, ckpt

__all__ = ['ArcMarginProduct','ArcFaceModel','arcface_forward_embeddings','build_arcface_model','load_arcface_checkpoint']


############################################
#         Pipeline                         #
############################################


def crop_expand_box(box, W, H, margin):
    
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    w2 = w * (1.0 + margin)
    h2 = h * (1.0 + margin)

    nx1 = int(round(cx - w2/2.0))
    ny1 = int(round(cy - h2/2.0))
    nx2 = int(round(cx + w2/2.0))
    ny2 = int(round(cy + h2/2.0))

    nx1 = max(0, min(W - 1, nx1))
    ny1 = max(0, min(H - 1, ny1))
    nx2 = max(0, min(W - 1, nx2))
    ny2 = max(0, min(H - 1, ny2))

    if nx2 <= nx1 + 1:
        nx2 = min(W - 1, nx1 + 2)
    if ny2 <= ny1 + 1:
        ny2 = min(H - 1, ny1 + 2)

    return [nx1, ny1, nx2, ny2]


class FacePipeline:
    def __init__(self, hg_ckpt_path, af_ckpt_path, device):

        self.device = torch.device(device)

        # загружаем модели
        self.hg_model, _ = load_hourglass_checkpoint(str(hg_ckpt_path), device=self.device)
        self.arc_model, _ = load_arcface_checkpoint(str(af_ckpt_path), device=self.device)

        # определяем детектор
        self.detector = MTCNN(keep_all=True, device=self.device)

    @torch.no_grad()
    def infer(self, img_pil, margin=-0.05, det_thr=0.90, batch_size=16):

        # загружаем картинку
        img = img_pil.convert("RGB")
        W, H = img.size

        # детектируем лица на картинке
        boxes, probs = self.detector.detect(img)

        # фильтруем боксы по порогу det_thr, расширяем/уменьшаем по margin 
        kept = []
        for box, p in zip(boxes, probs):
            if p is None or float(p) < float(det_thr):
                continue
            b = crop_expand_box(box.tolist(), W, H, margin=margin)
            kept.append((b, float(p)))

        # делаем кропы и ресайз до 256x256 для hourglass
        faces_256_pil = []
        faces_256 = []
        for b, _p in kept:

            x1, y1, x2, y2 = map(int, b)
            crop = img.crop((x1, y1, x2, y2))

            face256 = crop.resize((256, 256), Image.BILINEAR)
            faces_256_pil.append(face256)
            faces_256.append(T.ToTensor()(face256))

        faces_256 = torch.stack(faces_256, 0).to(self.device)

        # предсказаем ключевые точки по батчам
        pts_all = []
        for i in range(0, len(kept), batch_size):
            pts = predict_points_batch(self.hg_model, faces_256[i:i+batch_size], device=self.device,
                                       hm_size=64, input_size=256)
            pts_all.append(pts)
        pts_all = np.concatenate(pts_all, axis=0)

        # выровниваем лица по ключевым точкам
        aligned_256_pil = []
        for f_pil, pts in zip(faces_256_pil, pts_all):
            aligned = align_face_affine(np.array(f_pil), pts, output_size=256)
            aligned_256_pil.append(Image.fromarray(aligned).convert("RGB"))

        # делаем ресайз до 224x224 для arcface
        faces_224 = torch.stack([T.ToTensor()(p.resize((224, 224), Image.BILINEAR)) for p in aligned_256_pil], 0).to(self.device)

        # предсказываем эмбеддинги
        z = arcface_forward_embeddings(self.arc_model, faces_224).detach().cpu().numpy()

        out = []
        for (b, p), emb in zip(kept, z):
            out.append({"box": b, "embedding": emb})

        # возвращаем bboxы и эмбеддинги
        return out