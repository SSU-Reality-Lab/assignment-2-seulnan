import os, cv2, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import features

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def save_heatmap(array, title, filename, cmap='jet'):
    plt.imshow(array, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    os.makedirs('results', exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def save_keypoints(image, keypoints, filename):
    vis = image.copy()
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(vis, (x, y), 2, (0,255,0), -1)
    os.makedirs('results', exist_ok=True)
    cv2.imwrite(filename, vis)

# -------------------------------------------------------------------
# 0️⃣ Load Images
# -------------------------------------------------------------------
img1 = cv2.imread('resources/yosemite1.jpg')
img2 = cv2.imread('resources/yosemite2.jpg')

gray1 = cv2.cvtColor(img1.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2.astype(np.float32)/255.0, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 1️⃣ Feature Computation (TODO1~6)
# -------------------------------------------------------------------
HKD = features.HarrisKeypointDetector()
SFD = features.SimpleFeatureDescriptor()
MFD = features.MOPSFeatureDescriptor()

# TODO1
a1, b1 = HKD.computeHarrisValues(gray1)
a2, b2 = HKD.computeHarrisValues(gray2)

# TODO3
d1 = HKD.detectKeypoints(img1)
d2 = HKD.detectKeypoints(img2)

# Filter weak keypoints
d1 = [kp for kp in d1 if kp.response > 0.01]
d2 = [kp for kp in d2 if kp.response > 0.01]

# TODO4~6
desc_simple_1 = SFD.describeFeatures(img1, d1)
desc_simple_2 = SFD.describeFeatures(img2, d2)
desc_mops_1 = MFD.describeFeatures(img1, d1)
desc_mops_2 = MFD.describeFeatures(img2, d2)

# -------------------------------------------------------------------
# 2️⃣ Visualization (TODO1, TODO3)
# -------------------------------------------------------------------
save_heatmap(a1, "Image1 - TODO1 Harris Strength", "results/img1_TODO1_harris_strength.png")
save_heatmap(a2, "Image2 - TODO1 Harris Strength", "results/img2_TODO1_harris_strength.png")

save_keypoints(img1, d1, "results/img1_TODO3_detected_keypoints.png")
save_keypoints(img2, d2, "results/img2_TODO3_detected_keypoints.png")

print("✅ Saved TODO1 & TODO3 visualizations.")

# -------------------------------------------------------------------
# 3️⃣ Matching (TODO7 - SSD, TODO8 - Ratio)
# -------------------------------------------------------------------
matcher_ssd = features.SSDFeatureMatcher()
matcher_ratio = features.RatioFeatureMatcher()

# ------------------------------
# TODO7 - SSD matching
# ------------------------------
# Step 1. SSD matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ssd = matcher_ssd.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. 거리(distance)를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ssd = sorted(matches_ssd, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ssd_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ssd, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO7_SSD_matches.png", ssd_vis)
print("✅ TODO7 (SSD) match result saved → results/TODO7_SSD_matches.png")

# ------------------------------
# TODO8 - Ratio matching
# ------------------------------
# Step 1. Ratio matcher를 이용해 두 이미지의 MOPS 디스크립터 매칭을 수행하시오.
matches_ratio = matcher_ratio.matchFeatures(desc_mops_1, desc_mops_2)

# Step 2. distance를 기준으로 정렬하여 상위 150개의 매칭만 선택하시오.
matches_ratio = sorted(matches_ratio, key=lambda x: x.distance)[:150]

# Step 3. 매칭 결과를 시각화하여 PNG로 저장하시오.
ratio_vis = cv2.drawMatches(
    img1, d1, img2, d2, matches_ratio, None,
    matchColor=(0,255,0), singlePointColor=(255,0,0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)
cv2.imwrite("results/TODO8_Ratio_matches.png", ratio_vis)
print("✅ TODO8 (Ratio) match result saved → results/TODO8_Ratio_matches.png")

print("🎯 All TODO1–8 visualizations done! Files saved in 'results/'")

'''
---------------------------------------------------------------
[각주] TODO 8의 매칭이 TODO 7보다 더 잘되는 이유

- SSD 매칭(TODO 7)은 각 특징에 대해 단순히 가장 작은 SSD를 갖는 짝을 선택하므로,
  텍스처가 반복적이거나 배경 패턴이 유사한 영역에서 오매칭이 쉽게 발생한다.
- Ratio 매칭(TODO 8)은 features.RatioFeatureMatcher.matchFeatures에서
  최단거리 SSD1과 차선 SSD2의 비(SSD1/SSD2)를 distance로 사용한다.
  SSD1과 SSD2가 비슷하면(=모호한 매칭) 비율이 1에 가까워지고, 확실한 매칭일수록 비율이 작아진다.
  따라서 모호한 후보를 자동으로 낮은 우선순위로 밀어내어 잘못된 매칭을 효과적으로 억제한다.
- 본 프로젝트의 MOPS 디스크립터는 회전·스케일 정규화를 통해 패치 비교를 안정화하지만,
  여전히 반복 패턴/저대비 영역에서는 여러 후보가 비슷한 SSD를 낼 수 있다.
  이때 Ratio Test가 두 번째 후보와의 간격을 반영해 신뢰도 높은 대응만 상위로 남기므로
  실질적인 매칭 품질이 SSD 단독 기준보다 좋아진다.
'''
