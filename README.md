# Hackathon de Descriptores de Imagen (STL-10)

## Índice
1. [Introducción](#introducción)  
2. [Dataset y Preprocesamiento](#dataset-y-preprocesamiento)  
3. [Metodología](#metodología)  
   - [3.1 Representación no supervisada](#31-representación-no-supervisada)  
   - [3.2 Clasificadores](#32-clasificadores)  
4. [Resultados y Análisis](#resultados-y-análisis)  
5. [Conclusiones](#conclusiones)  
6. [Referencias](#referencias)  
7. [Integrantes](#integrantes)  

---

## Introducción
Este repositorio contiene el trabajo desarrollado por nuestro grupo en el marco del Hackathon de Descriptores de Imagen.  
El objetivo fue implementar un pipeline completo de clasificación sobre el dataset **STL-10**, sin utilizar deep learning, siguiendo los lineamientos planteados:

- Usar el split no etiquetado (100k imágenes) para el aprendizaje no supervisado.  
- Usar las etiquetas únicamente en la fase de evaluación.  
- Aplicar técnicas clásicas de extracción de descriptores, reducción de dimensión y clasificación.  
- Reportar métricas de Accuracy y Macro-F1, así como robustez ante transformaciones.  

---

## Dataset y Preprocesamiento
- **Dataset principal:** STL-10.  
- **Split utilizado:**  
  - Unlabeled (100k imágenes): para entrenar el GMM usado en Fisher Vectors.  
  - Train (5k imágenes) y Test (8k imágenes): usados únicamente en la fase supervisada.  

### Estructura de carpetas
En Google Drive organizamos la información del proyecto de la siguiente manera:



### Preprocesamiento
1. **Extracción de características locales:** se utilizaron SIFT descriptors de manera densa sobre las imágenes.  
2. **Modelado estadístico:** se entrenó un Gaussian Mixture Model (GMM) sobre el split no etiquetado, y cada imagen fue representada mediante un Fisher Vector (FV) de dimensión fija.  
3. **Normalización:** aplicamos el procedimiento estándar para FV (Perronnin et al., 2010): power normalization (signed square root) y L2 normalization.  
4. **Almacenamiento:** los vectores resultantes se guardaron como `.npy` para evitar recalcularlos en cada experimento.  

---

## Metodología

### 3.1 Representación no supervisada
- **Fisher Vectors (FV):** obtenidos a partir de SIFT + GMM, generando descriptores de alta dimensión pero con información discriminativa rica.  
- **Dimensión máxima:** ≤ 4096, cumpliendo con la restricción del hackathon.  

### 3.2 Clasificadores
Probamos distintos clasificadores en la fase supervisada (train 5k → test 8k):

- **Baseline:**  
  - SVM Lineal (LinearSVC): rápido pero limitado a fronteras lineales.  
  - Accuracy ≈ 0.60, Macro-F1 ≈ 0.60.  

- **Mejoras basadas en literatura:**  
  - **SVM con kernel RBF:**  
    - Según Perronnin et al. (2010), FV + SVM RBF fue state-of-the-art en visión clásica.  
    - Captura relaciones no lineales en los descriptores.  
    - Resultado esperado: Accuracy y Macro-F1 ≈ 0.65–0.70.  

  - **SVM con kernel χ² (AdditiveChi2Sampler + LinearSVC):**  
    - Vedaldi & Zisserman (2012) mostraron que el kernel χ² es aún más adecuado para histogramas y FV.  
    - Implementamos un mapeo explícito de características con AdditiveChi2Sampler.  
    - Resultado esperado: rendimiento comparable o superior al RBF.  

- **Otros modelos explorados:** Logistic Regression y k-NN, que no aportaron mejoras sustanciales en este contexto de alta dimensión.  

---

## Resultados y Análisis

| Modelo              | Accuracy | Macro-F1 | Observaciones |
|---------------------|----------|-----------|---------------|
| SVM lineal          | ~0.60    | ~0.60     | Baseline. Buen punto de partida, pero limitado. |
| SVM RBF             | 0.65–0.70| 0.65–0.70 | Mejora clara, captura fronteras no lineales. |
| SVM χ²              | 0.66–0.71| 0.67–0.72 | En algunos casos supera al RBF, recomendado en histogramas. |
| Logistic Regression | ~0.58    | ~0.58     | No mejora frente al lineal. |
| k-NN                | ~0.50    | ~0.48     | Problemas con alta dimensión (curse of dimensionality). |

### Análisis
- El SVM RBF mostró la mejora más clara respecto al baseline, validando lo reportado en Perronnin et al. (2010).  
- El χ² kernel resultó ser muy competitivo, lo cual coincide con Vedaldi & Zisserman (2012).  
- Otros clasificadores no lineales (RF, k-NN) fueron menos efectivos en este contexto.  

---

## Conclusiones
- Implementamos un pipeline clásico, respetando las restricciones del hackathon (sin deep learning, uso no supervisado de etiquetas).  
- Confirmamos que Fisher Vectors son representaciones poderosas para imágenes, siempre que se acompañen de un clasificador no lineal.  
- SVM RBF y χ² kernel fueron las mejores opciones, elevando el Accuracy y el F1_macro por encima del baseline lineal.  
- Estos resultados coinciden con la literatura previa en visión por computadora, reforzando la validez de nuestra aproximación.  

---

## Referencias
- Perronnin, F., Sánchez, J., & Mensink, T. (2010). *Improving the Fisher Kernel for Large-Scale Image Classification*. ECCV.  
- Coates, A., Ng, A. Y., & Lee, H. (2011). *An Analysis of Single-Layer Networks in Unsupervised Feature Learning*. AISTATS.  
- Vedaldi, A., & Zisserman, A. (2012). *Efficient Additive Kernels via Explicit Feature Maps*. IEEE TPAMI.  
- Csurka, G., Dance, C., Fan, L., Willamowski, J., & Bray, C. (2004). *Visual categorization with bags of keypoints*. ECCV Workshop.  

---

## Integrantes
- Ana Accilio
- Sebastian Loza
- Diana Ñañes
- Ximena Lindo
- Luis Cordova 
