import streamlit as st
import numpy as np
import os
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
import matplotlib.cm as cm  # 히트맵 컬러맵 변환용

# ── 상수 및 경로 설정 ──
INPUT_IMG_SIZE = (224, 224)
NEG_CLASS = 1
CLASSES = ["정상", "불량"]
MODEL_PATH = "./weights/leather_model.keras"
HEATMAP_THRES = 0.5

# ─────────────────────────────────────────────
# 1. 페이지 설정 (제목, 아이콘, 레이아웃)
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly - 가죽 이상 탐지", page_icon="📷", layout="wide")

st.title("InspectorsAlly")
st.caption("AI 기반 자동 비전 검사로 제조 품질 관리를 한 단계 높이세요")
st.write("가죽 표면 이미지를 업로드하거나 카메라로 촬영하면 AI 모델이 **정상 / 불량** 여부를 실시간으로 판별합니다.")

# 사이드바 대시보드 구성
with st.sidebar:
    if os.path.exists("./docs/overview_dataset.jpg"):
        st.image(Image.open("./docs/overview_dataset.jpg"), use_container_width=True)
    st.subheader("InspectorsAlly 소개")
    st.write(
        "InspectorsAlly는 기업의 품질 관리 검사를 효율화하기 위해 설계된 "
        "AI 기반 비전 검사 애플리케이션입니다. VGG16 전이학습 기법을 활용하여 "
        "가죽 제품의 스크래치, 찍힘, 오염 등의 결함을 미세하게 감지합니다."
    )
    st.divider()
    st.write("**시스템 및 모델 정보**")
    st.write(f"- 프레임워크: TensorFlow {tf.__version__}")
    st.write(f"- 백본 네트워크: VGG16 (ImageNet 사전학습 가중치 활용)")
    st.write(f"- 활성화 함수 및 출력: Sigmoid 이진 출력 (0: 정상, 1: 불량)")
    st.write(f"- 입력 해상도: {INPUT_IMG_SIZE[0]} × {INPUT_IMG_SIZE[1]} 픽셀")

# ─────────────────────────────────────────────
# 2. 모델 로드 및 캐싱 (@st.cache_resource)
# ─────────────────────────────────────────────
@st.cache_resource
def load_leather_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    try:
        # 구조와 가중치 통합 본 로드
        model = tf.keras.models.load_model(MODEL_PATH)

        # CAM 히트맵 분석용 서브네트워크(cam_model) 재구성
        vgg16 = model.get_layer("vgg16")
        inputs = vgg16.input
        feature_out = vgg16.get_layer("block5_conv3").output

        x = vgg16.output
        x = model.get_layer("global_average_pooling2d")(x)
        x = model.get_layer("dense")(x)
        x = model.get_layer("dropout")(x)
        predictions = model.get_layer("predictions")(x)

        cam_model = keras.Model(inputs=inputs, outputs=[feature_out, predictions])
        return model, cam_model
    except Exception as e:
        return None, None

# 모델 로딩 및 예외 처리
model, cam_model = load_leather_model()

if model is None:
    st.error(
        f"🚨 **모델 파일을 로드할 수 없습니다.** 경로를 확인해 주세요.\n\n"
        f"- 예상 경로: `{MODEL_PATH}`\n"
        f"- 해결 방법: 해당 경로에 가중치가 포함된 유효한 `.keras` 또는 `.h5` 파일을 배치하세요."
    )
    st.stop()


# ─────────────────────────────────────────────
# 3. 이미지 전처리 함수
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    img_array = keras.applications.vgg16.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


# ─────────────────────────────────────────────
# 4. CAM 히트맵 및 바운딩 박스 연산
# ─────────────────────────────────────────────
def generate_heatmap(cam_model, img_array):
    feature_maps, pred = cam_model(img_array, training=False)
    feature_maps = feature_maps.numpy()[0]
    prob = float(pred.numpy()[0][0])
    class_idx = 1 if prob > HEATMAP_THRES else 0

    w1 = cam_model.get_layer("dense").get_weights()[0]
    w2 = cam_model.get_layer("predictions").get_weights()[0]
    weights_for_anomaly = (w1 @ w2).squeeze()

    cam = np.dot(feature_maps, weights_for_anomaly)
    cam_min, cam_max = cam.min(), cam.max()
    norm_cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    heatmap_pil = Image.fromarray((norm_cam * 255).astype(np.uint8))
    heatmap_resized = np.array(heatmap_pil.resize(INPUT_IMG_SIZE)) / 255.0
    return heatmap_resized, prob, class_idx


def get_bbox_from_heatmap(heatmap, thres=0.5):
    binary_map = heatmap > thres
    if not binary_map.any():
        return None
    x_dim = np.max(binary_map, axis=0) * np.arange(binary_map.shape[1])
    y_dim = np.max(binary_map, axis=1) * np.arange(binary_map.shape[0])
    x_vals = x_dim[x_dim > 0]
    y_vals = y_dim[y_dim > 0]
    if len(x_vals) == 0 or len(y_vals) == 0:
        return None
    return int(x_vals.min()), int(y_vals.min()), int(x_dim.max()), int(y_dim.max())


# ─────────────────────────────────────────────
# 5. 결과 시각화 (Matplotlib 제거 -> st.image 대체 완료)
# ─────────────────────────────────────────────
def visualize_result_st(pil_img, heatmap, class_idx, prob, thres=HEATMAP_THRES):
    img_np = np.array(pil_img.resize(INPUT_IMG_SIZE).convert("RGB"))
    
    if class_idx == NEG_CLASS:  # 불량일 경우 (시각화 분석 제공)
        # Matplotlib 연산 윈도우 없이 백엔드에서 직접 Reds 컬러맵 추출
        heatmap_colored = cm.Reds(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # 원본 이미지와 알파 블렌딩 오버레이 합성 (투명도 45%)
        alpha = 0.45
        blended = (alpha * heatmap_colored + (1 - alpha) * img_np).astype(np.uint8)
        blended_pil = Image.fromarray(blended)
        
        # 불량 바운딩 박스 감지 시 PIL 기법으로 사각형 그리기
        bbox = get_bbox_from_heatmap(heatmap, thres)
        if bbox:
            x0, y0, x1, y1 = bbox
            draw = ImageDraw.Draw(blended_pil)
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            
        # UI 레이아웃을 2단 컬럼으로 분할하여 병렬 시각화 출력
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.image(img_np, caption="원본 이미지", use_container_width=True)
        with viz_col2:
            st.image(blended_pil, caption=f"결함 집중부 분석 히트맵 (이상 수치: {prob:.3f})", use_container_width=True)
            
    else:  # 정상일 경우
        st.image(img_np, caption=f"양품 판정 표면 데이터", width=350)


# ─────────────────────────────────────────────
# 6. 사용자 입력 인터페이스 (업로드 vs 촬영 토글)
# ─────────────────────────────────────────────
st.write("---")
st.subheader("📥 검사 대상 이미지 입력")
input_method = st.radio("입력 방식을 선택하세요:", ["파일 업로드", "카메라 촬영"])

pil_image = None

if input_method == "파일 업로드":
    uploaded_file = st.file_uploader("가죽 샘플 이미지 파일을 선택하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="선택된 업로드 이미지 미리보기", width=350)
        st.success("✔ 파일 등록 완료!")

elif input_method == "카메라 촬영":
    camera_file = st.camera_input("정면에 가죽 부위를 맞춘 후 촬영 버튼을 눌러주세요")
    if camera_file is not None:
        pil_image = Image.open(camera_file)
        st.success("✔ 카메라 스냅샷 등록 완료!")


# ─────────────────────────────────────────────
# 7. 검사 컨트롤러 및 실시간 결과 대시보드
# ─────────────────────────────────────────────
st.write("---")
submit = st.button(label="🔍 가죽 불량 검사 시작", type="primary", use_container_width=True)

if submit:
    # 예외 상황 처리: 데이터가 비어있을 때 경고 안내
    if pil_image is None:
        st.warning("⚠️ 분석할 이미지 데이터가 존재하지 않습니다. 파일을 먼저 업로드하거나 카메라 촬영을 완료해 주세요.")
    else:
        st.subheader("💡 실시간 품질 검사 판정 결과")
        
        with st.spinner("AI 딥러닝 알고리퍼가 결함 영역을 추론 및 전처리 중입니다..."):
            # 입력 데이터 전처리 및 추론 파이프라인 가동
            img_array = preprocess_image(pil_image)
            heatmap, prob, class_idx = generate_heatmap(cam_model, img_array)
            
        # 요구사항 반영: 정상이면 success, 불량이면 error 상태창 표기
        if class_idx == 0:
            st.success(f"✅ **최종 결과: 정상 (합격품)** — 제품 품질 기준 조건을 통과하였습니다.")
        else:
            st.error(f"❌ **최종 결과: 불량 감지 (불합격품)** — 미세 결함 및 이상 징후 영역이 탐지되었습니다.")
            
        # 요구사항 반영: st.metric으로 정상/불량 비율 나란히 병렬 배치
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(label="📊 정상 판정 확률", value=f"{(1 - prob):.1%}")
        metric_col2.metric(label="🚨 불량 발생 확률", value=f"{prob:.1%}", 
                           delta=f"{prob:.1%}" if prob > HEATMAP_THRES else None, delta_color="inverse")
        
        # 요구사항 반영: st.progress로 불량 위험도 막대 그래프 표시
        st.write("**불량 위험도 진행 수준**")
        st.progress(float(prob), text=f"위험 지수 계측치: {prob:.1%}")
        
        # 요구사항 반영: Matplotlib 제거 후 st.image 조합으로 시각화 출력
        st.write("---")
        st.write("#### 🗺️ 결함 하이라이트 매핑 (Grad-CAM 분석)")
        visualize_result_st(pil_image, heatmap, class_idx, prob, thres=HEATMAP_THRES)