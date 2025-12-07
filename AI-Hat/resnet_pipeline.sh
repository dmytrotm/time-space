#!/bin/bash
###############################################################################
# Pipeline для ResNet18 на Hailo AI HAT+
# PyTorch -> ONNX -> HEF -> Inference
###############################################################################

set -e

# Кольори
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
warning() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
step() { echo -e "${CYAN}[STEP]${NC} $1"; }

echo ""
echo "╔════════════════════════════════════════╗"
echo "║   ResNet18 Hailo Pipeline              ║"
echo "║   PyTorch → ONNX → HEF → Inference     ║"
echo "╚════════════════════════════════════════╝"
echo ""

# Перевірка середовища
check_environment() {
    step "1/6: Перевірка середовища"
    
    # Hailo CLI
    if ! command -v hailortcli &> /dev/null; then
        error "hailortcli не знайдено!"
        exit 1
    fi
    success "hailortcli: $(hailortcli --version 2>&1 | head -1)"
    
    # Python
    if ! command -v python &> /dev/null; then
        error "Python 3 не знайдено!"
        exit 1
    fi
    success "Python: $(python --version)"
    
    # Hailo пристрій
    if sudo hailortcli scan 2>/dev/null | grep -q "Hailo"; then
        success "Hailo пристрій виявлено"
        sudo hailortcli scan | grep "Device:"
    else
        warning "Hailo пристрій не виявлено (продовжуємо...)"
    fi
    
    echo ""
}

# Створення структури
setup_structure() {
    step "2/6: Створення структури директорій"
    
    mkdir -p models/onnx
    mkdir -p models/hef
    mkdir -p images/test
    mkdir -p images/calib
    mkdir -p results
    
    success "Структура створена"
    tree -L 2 models 2>/dev/null || ls -R models
    echo ""
}

# Конвертація PyTorch -> ONNX
pytorch_to_onnx() {
    step "3/6: Конвертація PyTorch -> ONNX"
    
    if [ ! -f "AI-Hat/resnet2onnx.py" ]; then
        warning "scripts/resnet2onnx.py не знайдено"
        info "Скопіюй свій resnet2onnx.py в AI-Hat/"
        return 1
    fi
    
    info "Запуск resnet2onnx.py..."
    cd AI-Hat
    python resnet2onnx.py
    cd ..
    
    # Перевірка
    if [ -f "models/onnx/resnet18.onnx" ]; then
        local size=$(du -h models/onnx/resnet18.onnx | cut -f1)
        success "ONNX створено: models/onnx/resnet18.onnx ($size)"
    else
        error "ONNX не створено!"
        return 1
    fi
    
    echo ""
}

# Конвертація ONNX -> HEF
onnx_to_hef() {
    step "4/6: Конвертація ONNX -> HEF"
    
    local onnx_path="models/onnx/resnet18.onnx"
    local har_path="models/onnx/resnet18.har"
    local hef_path="models/hef/resnet18.hef"
    
    if [ ! -f "$onnx_path" ]; then
        error "ONNX не знайдено: $onnx_path"
        return 1
    fi
    
    # Parse
    info "Parsing ONNX -> HAR..."
    hailo parser onnx "$onnx_path" --hw-arch hailo8l
    
    if [ ! -f "$har_path" ]; then
        error "HAR не створено після parsing"
        return 1
    fi
    success "HAR створено"
    
    # Optimize
    if [ -d "images/calib" ] && [ "$(ls -A images/calib 2>/dev/null)" ]; then
        local num_images=$(ls images/calib | wc -l)
        info "Оптимізація з калібраційними даними ($num_images зображень)..."
        hailo optimizer "$har_path" \
            --calib-dataset images/calib \
            --output-file "${har_path%.har}_optimized.har"
        har_path="${har_path%.har}_optimized.har"
        success "Оптимізація з калібрацією завершена"
    else
        warning "Калібраційні дані не знайдено в images/calib/"
        info "Оптимізація без калібрації..."
        hailo optimizer "$har_path"
        success "Оптимізація завершена"
    fi
    
    # Compile
    info "Компіляція HAR -> HEF..."
    hailo compiler "$har_path" -o "$hef_path"
    
    if [ -f "$hef_path" ]; then
        local size=$(du -h "$hef_path" | cut -f1)
        success "HEF створено: $hef_path ($size)"
    else
        error "HEF не створено!"
        return 1
    fi
    
    echo ""
}

# Тестовий inference
test_inference() {
    step "5/6: Тестовий inference"
    
    local hef_path="models/hef/resnet18.hef"
    
    if [ ! -f "$hef_path" ]; then
        error "HEF не знайдено: $hef_path"
        return 1
    fi
    
    # Перевірка тестових зображень
    if [ ! "$(ls -A images/calib 2>/dev/null)" ]; then
        warning "Тестові зображення не знайдено в images/calib/"
        warning "Пропускаємо тестування"
        return 0
    fi
    
    local test_image=$(ls images/calib/* | head -1)
    
    if [ ! -f "AI-Hat/resnet_inference.py" ]; then
        warning "AI-Hat/resnet_inference.py не знайдено"
        warning "Пропускаємо тестування"
        return 0
    fi
    
    info "Запуск inference на $test_image..."
    python3 scripts/resnet_inference.py \
        --model "$hef_path" \
        --input "$test_image" \
        --output "results/test_result.jpg" \
        --threshold 0.5
    
    if [ -f "results/test_result.jpg" ]; then
        success "Результат збережено: results/test_result.jpg"
    fi
    
    echo ""
}

# Benchmark
benchmark_model() {
    step "6/6: Benchmark"
    
    local hef_path="models/hef/resnet18.hef"
    
    if [ ! -f "$hef_path" ]; then
        warning "HEF не знайдено, пропускаємо benchmark"
        return 0
    fi
    
    info "Запуск benchmark (10 секунд)..."
    echo ""
    sudo hailortcli benchmark "$hef_path" --time 10
    echo ""
    success "Benchmark завершено"
    echo ""
}

# Головна функція
main() {
    local start_time=$(date +%s)
    
    check_environment
    setup_structure
    
    # Pipeline
    if pytorch_to_onnx; then
        if onnx_to_hef; then
            test_inference || true
            benchmark_model || true
        fi
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    echo "╔════════════════════════════════════════╗"
    echo "║         Pipeline завершено! ✅         ║"
    echo "╚════════════════════════════════════════╝"
    echo ""
    info "Загальний час: ${duration}s"
    echo ""
    
    # Показати результати
    if [ -f "models/hef/resnet18.hef" ]; then
        success "Модель готова: models/hef/resnet18.hef"
        echo ""
        info "Наступні кроки:"
        echo ""
        echo "  # Inference на зображенні:"
        echo "  python3 AI-Hat/resnet_inference.py \\"
        echo "      --model models/hef/resnet18.hef \\"
        echo "      --input test.jpg --show"
        echo ""
        echo "  # Batch обробка:"
        echo "  python3 AI-Hat/resnet_inference.py \\"
        echo "      --model models/hef/resnet18.hef \\"
        echo "      --batch images/test/ --output results/"
     
    else
        error "HEF не створено. Перевір логи вище."
    fi
}

# Обробка аргументів
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Використання: $0"
    echo ""
    echo "Pipeline автоматично:"
    echo "  1. Перевіряє середовище"
    echo "  2. Створює структуру директорій"
    echo "  3. Конвертує PyTorch -> ONNX (через resnet2onnx.py)"
    echo "  4. Конвертує ONNX -> HEF (Hailo format)"
    echo "  5. Тестує inference"
    echo "  6. Запускає benchmark"
    echo ""
    echo "Перед запуском:"
    echo "  - Помісти resnet2onnx.py в scripts/"
    echo "  - Помісти resnet_inference.py в scripts/"
    echo "  - (Опціонально) Додай калібраційні зображення в images/calib/"
    echo ""
    exit 0
fi

# Запуск
main