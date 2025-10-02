# Informe del Hackathon – Clasificación de Imágenes con Descriptores Clásicos y SVM

---

## Índice
1. Introducción  
2. Dataset y Preprocesamiento  
3. Extracción de Descriptores  
   - Fisher Vectors  
   - Gaussian Mixture Models (GMM)  
   - Reducción de dimensionalidad (opcional)  
4. Modelos de Clasificación  
   - SVM Lineal  
   - SVM con Kernel RBF  
   - Optimización de Hiperparámetros  
   - Exploración de Kernel χ²  
5. Resultados  
6. Discusión  
7. Conclusiones  
8. Integrantes  
9. Referencias  

---

## 1. Introducción
El objetivo del hackathon fue **clasificar imágenes del dataset STL-10** utilizando únicamente **técnicas clásicas de visión por computador**, evitando el uso de redes neuronales profundas.  
Nuestro grupo planteó un pipeline basado en:  

- **Descriptores Fisher Vectors (FV)** para representar imágenes.  
- **Clasificadores supervisados simples (SVM, k-NN, Logistic Regression)** para la evaluación.  
- **Validación cruzada y búsqueda de hiperparámetros** para mejorar la precisión y la robustez de los modelos.  

---

## 2. Dataset y Preprocesamiento
- **Dataset STL-10**:  
  - 10 clases (avión, pájaro, auto, etc.).  
  - 5,000 imágenes de entrenamiento (etiquetadas).  
  - 8,000 imágenes de test.  
  - 100,000 imágenes no etiquetadas para representación no supervisada.  

- **Preprocesamiento**:  
  1. Extracción de descriptores locales en cada imagen.  
  2. Entrenamiento de un **Gaussian Mixture Model (GMM)** sobre los descriptores locales.  
  3. Representación de cada imagen como un **Fisher Vector (FV)** derivado del GMM.  
  4. Guardado en formato `.npy`:  
     - `Xtr_fv.npy` → Fisher Vectors de entrenamiento.  
     - `Xte_fv.npy` → Fisher Vectors de test.  
     - `ytr.npy`, `yte.npy` → etiquetas correspondientes.  

Esto evitó recalcular descriptores cada vez, optimizando el tiempo de experimentación.  

---

## 3. Extracción de Descriptores

### Fisher Vectors (FV)
Los **Fisher Vectors** son una representación avanzada que codifica cómo los descriptores locales de una imagen se desvían de un modelo generativo global (en este caso, un GMM).  

Formalmente, dado un GMM con parámetros \(\lambda = \{w_k, \mu_k, \Sigma_k\}_{k=1}^K\), el FV de una imagen es un vector que combina:  
- Derivadas respecto a las medias \(\mu_k\).  
- Derivadas respecto a las covarianzas \(\Sigma_k\).  

Esto captura información **de primer y segundo orden** sobre la distribución de los descriptores locales.  

---

### Gaussian Mixture Model (GMM)
- Entrenamos un **GMM** con \(K\) gaussianas sobre un conjunto de descriptores locales extraídos de imágenes no etiquetadas.  
- Cada gaussiana modela un “cluster” de patrones visuales.  
- Los FV se construyen al medir cuánto contribuye cada imagen a cada gaussiana.  

> Ejemplo: un FV de dimensión 4096 puede representar la desviación de una imagen respecto a 64 gaussianas.  

---

### Reducción de dimensionalidad
Dado que los FV son de muy alta dimensión:  
- Se aplicó **PCA (Principal Component Analysis)** en algunos experimentos para reducir la dimensión y acelerar los clasificadores.  
- Limitación: no más de 4096 dimensiones finales (regla del hackathon).  

---

## 4. Modelos de Clasificación

### 4.1 SVM Lineal
- Modelo: `LinearSVC(loss="hinge")`.  
- Búsqueda sobre el hiperparámetro `C`.  
- Resultado óptimo:  
  - `C=1`  
  - Accuracy ≈ **0.6072**  
  - F1_macro ≈ **0.605**  

Esto sirvió como **baseline** inicial.  

---

### 4.2 SVM con Kernel RBF
- Modelo: `SVC(kernel="rbf")`.  
- Primer experimento con parámetros por defecto (`C=10`, `gamma="scale"`) dio:  
  - Accuracy ≈ **0.6230**  
  - F1_macro ≈ **0.6221**  

Esto ya superó el baseline lineal.  

📖 Esto coincide con la literatura: los FV funcionan mejor con kernels no lineales porque preservan la estructura geométrica de los descriptores (Perronnin et al., ECCV 2010).  

---

### 4.3 Optimización de Hiperparámetros
Se implementó un **GridSearch** manual sobre los hiperparámetros:  

- `C ∈ {0.1, 1, 3, 10}`  
- `gamma ∈ {"scale", 1e-3, 3e-3, 1e-2}`  

Validación cruzada estratificada con 3 folds.  

- **Mejor configuración**: `C=3, gamma="scale"`  
- CV-acc ≈ **0.6016**  
- Accuracy en test ≈ **0.62+**  

---

### 4.4 Exploración de Kernel χ²
- Se intentó `AdditiveChi2Sampler + LinearSVC`.  
- Sin embargo, los FV contienen **valores negativos**, mientras que el kernel χ² requiere histogramas no negativos.  
- Resultado: el modelo falló a menos que se aplicaran transformaciones (shift/abs).  
- Conclusión: el kernel χ² es más adecuado para descriptores histogramales (ej. **BoVW**, **LBP**).  

---

## 5. Resultados

| Modelo                    | Accuracy | F1-macro |
|----------------------------|----------|----------|
| **SVM Lineal (C=1)**       | 0.6072   | ~0.605   |
| **SVM RBF (C=10)**         | 0.6230   | 0.6221   |
| **SVM RBF (C=3, opt)**     | ~0.62+   | ~0.62+   |

---

## 6. Discusión
- Los Fisher Vectors demostraron ser **efectivos y consistentes** para la representación de imágenes.  
- El **SVM RBF** fue superior al SVM lineal, validando lo esperado en literatura.  
- La optimización de hiperparámetros mediante GridSearch permitió alcanzar configuraciones más estables.  
- La exploración con χ² fue académicamente interesante pero no aplicable directamente sobre FV.  

---

## 7. Conclusiones
1. Los **Fisher Vectors** son adecuados para el STL-10, capturan información robusta de las imágenes.  
2. El **SVM RBF** optimizado superó el baseline lineal en Accuracy y F1.  
3. La búsqueda de hiperparámetros es crucial en kernels no lineales.  
4. No todos los descriptores son compatibles con todos los kernels (ejemplo: χ² no se ajusta a FV).  
5. Nuestro pipeline logró **mejorar progresivamente** desde un baseline sencillo hasta un modelo más robusto y competitivo.  

---


## Integrantes
- Ana Accilio
- Sebastian Loza
- Diana Ñañes
- Ximena Lindo
- Luis Golac

  
----

## 9. Referencias
- Perronnin, F., Sánchez, J., & Mensink, T. (2010). *Improving the Fisher Kernel for Large-Scale Image Classification*. ECCV.  
- Zhang, J., Marszalek, M., Lazebnik, S., & Schmid, C. (2007). *Local Features and Kernels for Classification of Texture and Object Categories*. IJCV.  
- Coates, A., Ng, A., & Lee, H. (2011). *An Analysis of Single-Layer Networks in Unsupervised Feature Learning*. AISTATS.  

