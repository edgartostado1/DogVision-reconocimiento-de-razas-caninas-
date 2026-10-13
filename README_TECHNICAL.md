# EcoVision - Documentación Técnica (Resumen)

> **CNN Manual (79%, CPU, 94°C) → ResNet18 Fine-tuned (93.5%, GPU, 45°C)**

---

##  Resultado Final
| Métrica | Valor |
|---------|-------|
| **Val Accuracy** | **93.50%** |
| **Plástico** | 93.8% |
| **Vidrio** | 94.1% |
| **Papel** | 97.5% |
| **Cartón** | 93.8% |
| **Metal** | 87.8% |
| **Basura** | 89.3% |

---

##  Evolución Rápida

| Fase | Modelo | Device | Épocas | Accuracy |
|------|--------|--------|--------|----------|
| **Inicial** | CNN manual (7 clases) | CPU | 20 | ~79% |
| **Fix clases** | CNN manual (6 clases) | CPU | 15 | 79% |
| **Transfer Learning** | ResNet18 freeze | CPU | 15 | 79% |
| **GPU + Fine-tune** | ResNet18 unfreeze | **RTX 4050** | 10 | **93.5%** |

---

##  Cambios Clave

### 1. Arquitectura: CNN → ResNet18 Fine-tuned
```python
# ANTES: CNN casera 12M params from scratch
# AHORA: ResNet18 preentrenado ImageNet + fine-tuning completo
modelo = models.resnet18(weights=IMAGENET1K_V1)
modelo.fc = nn.Linear(512, 6)  # 6 clases
```

### 2. Bug Crítico: Orden de Clases
```python
# PROBLEMA: ImageFolder alfabético ≠ CATEGORIAS código
# FIX: CATEGORIAS = ["basura","carton","metal","papel","plastico","vidrio"]
# app.py: RECOMENDACIONES keys en minúsculas = CATEGORIAS
```

### 3. Configuración Fine-tuning (GPU)
```python
FREEZE_BACKBONE = False      # Entrena TODO
TASA_APRENDIZAJE = 0.0001    # LR bajo (crítico)
EPOCAS = 10
ReduceLROnPlateau(patience=3)
```

---

## 📊 Matriz Confusión - Plástico
| Real → Predicho | Cuenta |
|-----------------|--------|
| **plastico** | **91**  |
| metal | 2 |
| papel | 2 |
| vidrio | 2 |
| **carton** | **0**  |

---

##  Ejecución

```powershell
# Entrenar (GPU)
venv\Scripts\Activate.ps1
python -m modelo.entrenamiento  # ~3 min RTX 4050

# Demo
python app.py  # http://localhost:8501
```

---

##  Limitaciones
| Problema | Causa | Solución |
|----------|-------|----------|
| Vidrio ↔ Plástico transparente | 3% confusión | Más datos + augmentation |
| Basura (109 imgs) | Recall 89% | Class-weighted loss |
| Objetos OOD (perfumes) | Falla | Umbral confianza <60% |

---

##  Archivos Modificados
```
M modelo/red_neuronal.py      # ResNet18 + CATEGORIAS alfabético
M modelo/entrenamiento.py     # Fine-tuning + scheduler + guardado
M interfaz/app.py             # RECOMENDACIONES keys = CATEGORIAS
```

**Modelo final:** `modelo/modelo_entrenado.pth` (44.8 MB, 93.5%)

---
