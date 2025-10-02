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

