# Informe Técnico - Sistema MHealth HAR

Este directorio contiene el informe técnico del proyecto de Reconocimiento de Actividad Humana.

## Archivos

-   `informe.tex` - Documento principal en LaTeX (máx. 12 páginas)
-   `informe.pdf` - Documento principal compilado
-   `anexos.tex` - Material suplementario (código fuente, configuraciones)
-   `anexos.pdf` - Material suplementario compilado
-   `compile.sh` - Script para compilar los documentos (Linux/Mac)
-   `README.md` - Este archivo

## Compilación

### Requisitos

-   LaTeX completo (TeX Live, MiKTeX, o MacTeX)
-   Paquetes: babel, tikz, listings, hyperref, etc.

### Compilar en Linux/Mac

```bash
cd informe
chmod +x compile.sh
./compile.sh
```

### Compilar manualmente

```bash
cd informe
pdflatex informe.tex
pdflatex informe.tex  # Segunda pasada para referencias
```

### Compilar con latexmk (recomendado)

```bash
latexmk -pdf informe.tex
```

## Estructura del Documento

1. **Introducción** - Contexto, objetivos generales y específicos
2. **Dataset y Preprocesamiento** - MHealth, ventanas, features
3. **Arquitectura del Sistema** - Diagrama y estructura de carpetas
4. **Modelo de Machine Learning** - Random Forest, métricas
5. **Backend - API REST** - FastAPI, endpoints
6. **Frontend - Interfaz de Usuario** - React, funcionalidades
7. **Despliegue** - Docker, instrucciones
8. **Pruebas y Resultados** - Tests, validación end-to-end
9. **Uso de IA Generativa** - Prompts, iteraciones
10. **Reflexión y Mejoras Futuras** - MLOps, trabajo futuro
11. **Conclusiones**
12. **Anexos** - Configuración completa, enlace al repositorio

## Notas

-   El documento está limitado a 12 páginas sin contar anexos
-   El material suplementario (configuraciones, código ampliado) está en los anexos
-   Las figuras se generan con TikZ directamente en el documento
