#!/bin/bash

# Script para compilar el informe LaTeX
# Uso: ./compile.sh

echo "============================================"
echo "Compilando informe principal..."
echo "============================================"

# Informe principal - Primera pasada
pdflatex -interaction=nonstopmode informe.tex

# Informe principal - Segunda pasada (para referencias cruzadas y tabla de contenidos)
pdflatex -interaction=nonstopmode informe.tex

echo ""
echo "============================================"
echo "Compilando anexos..."
echo "============================================"

# Anexos - Primera pasada
pdflatex -interaction=nonstopmode anexos.tex

# Anexos - Segunda pasada
pdflatex -interaction=nonstopmode anexos.tex

# Limpiar archivos auxiliares (opcional, descomenta si deseas)
# rm -f *.aux *.log *.toc *.out *.lof *.lot

echo ""
echo "============================================"
echo "✅ Compilación completada."
echo "============================================"
echo "📄 Archivos generados:"
echo "   - informe.pdf (documento principal, máx 12 páginas)"
echo "   - anexos.pdf (material suplementario)"
