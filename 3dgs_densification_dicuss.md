**You:**

Uploaded a file
Phân tích bài báo sau đây. hãy tìm hiểu những problem setting dựa trên pipeline 3dgs (có thể tập trung vào cơ chế densification). Đầu tiên hãy tìm hiểu các bài liên quan và gần đây, phân tích xem họ đang làm gì, persecpective và angle của họ là gì, và các phương pháp gần đây đang cố gắng giải quyết vấn đề gì. Hãy tìm hiểu gap hiện tại chung của hướng này, người ta đang cố gắng giải quyết gì, mình có một phương pháp nào mới hơn dựa trên theory-based không để bài toán tôi áp dụng được novel hơn. Hãy liệt kê chi tiết những hướng đi phù hợp để nghiên cứu novel publish trên hội nghị A*. Hãy liệt kê càn chi tiết càng tốt và phân tích những math liên quan đến computer vision và neural repersentation. Phương pháp mới phải thay thế phương pháp hiện tại nhưng vẫn dựa trên những hướng đi của cộng đồng.

* * *

**ChatGPT:**

Dưới đây là một “bản đồ chiến trường” khá chi tiết về hướng **3D Gaussian Splatting (3DGS)** và đặc biệt là **densification / density control**, đặt trong bối cảnh bài bạn đưa (**Splat Regression Models**). Mình sẽ đi theo logic:

1. Bài “Splat Regression Models” (SRM) đang làm gì và vì sao nó _đụng đúng vào tim_ của densification
2. Pipeline 3DGS + densification truyền thống: cơ chế, math, và vì sao nó “heuristic”
3. Các bài gần đây đang sửa densification theo những “angle” nào, họ giải quyết vấn đề gì
4. Gap chung của hướng này
5. Gợi ý các hướng **theory-based** đủ “đắt” để push novelty (định hướng publish A*/top-tier), có phân tích math và cách “thay thế” cơ chế hiện tại nhưng vẫn đi đúng hướng cộng đồng

---

## 1) “Splat Regression Models” (SRM) đang làm gì, và nó liên quan densification ra sao?

Bài **Splat Regression Models** đưa một khung lý thuyết nhìn 3DGS/gaussian splatting như một trường hợp đặc biệt của một lớp mô hình rộng hơn: **mixture của các “bump functions” dị hướng** (splats), tối ưu hóa bằng **gradient flow trên không gian đo** (measure space) với hình học **Wasserstein–Fisher–Rao (WFR)**.

Splat Regression Models

Các điểm “đinh” (và cực liên quan densification):

* **Mô hình hóa bằng measure**: thay vì coi tham số là một vector hữu hạn chiều, SRM coi “tập splats” như một **mixing measure** trên không gian tham số của splat. Đây là cách nói rất đúng bản chất của 3DGS: bạn không chỉ tối ưu (b, Σ, màu) mà còn đang làm _model selection_ (thêm/bớt splat).

    Splat Regression Models

* **WFR = vừa “di chuyển khối lượng” (Wasserstein) vừa “teleport khối lượng” (Fisher–Rao)**. Trong ngôn ngữ 3DGS:
    * phần Wasserstein ~ cập nhật vị trí/hình dạng (move gaussians)
    * phần Fisher–Rao ~ cập nhật “khối lượng/weight” và gợi mở cơ chế **birth–death** (sinh–tử) một cách tự nhiên (đây chính là densification/pruning nếu làm đúng).
* SRM nói thẳng: họ muốn **tách pipeline thành các “module” rõ ràng** (inverse problem / model / optimization), và họ “recover 3D Gaussian Splatting như một instance”.
* Trong Example về NVS, họ viết NVS như một **bài toán inverse problem** với forward operator là **Radiative Transfer Equation (RTE)**, và 3DGS thực chất là một dạng xấp xỉ RTE bằng **α-blending** (một approximate renderer).

    Splat Regression Models

* Quan trọng cho densification: SRM cung cấp công thức **gradient WFR theo tham số splat** (Theorem 1), nghĩa là bạn có thể suy ra “ai đáng được tăng mật độ / ai đáng chết” bằng **first variation** δF.

Tóm lại: SRM đưa bạn một “lý thuyết nền” để biến densification từ **mẹo** thành **hệ quả của một bài toán tối ưu hóa trên measure**. Đây là mỏ vàng để làm paper kiểu “theory → algorithm → SOTA-ish improvement”.

---

## 2) Pipeline 3DGS và densification cổ điển: cơ chế + math (và vì sao nó dễ “rụng”)

### 2.1 Mô hình 3DGS (nhìn ở mức toán)

Một scene được biểu diễn bằng một tập Gaussians 3D dị hướng. Rendering: mỗi Gaussian được project sang 2D Gaussian trên ảnh; sau đó **alpha compositing (front-to-back)** để ra màu pixel. Nhiều paper (vd AbsGS) ghi rõ công thức kiểu này. [arXiv](https://arxiv.org/html/2404.10484v1)

### 2.2 Adaptive Density Control (ADC) gốc (densification + pruning)

ADC (mô tả rõ trong Revising Densification):

* Chọn Gaussian để “grow” nếu **độ lớn gradient vị trí** vượt threshold. [arXiv](https://arxiv.org/html/2404.06109v1)
* Nếu Gaussian “lớn” (dựa trên eigenvalue lớn nhất của covariance) thì **split**, còn nhỏ thì **clone**. [arXiv](https://arxiv.org/html/2404.06109v1)
* **Prune** nếu opacity thấp; thường kèm “opacity reset” để ép những cái không dùng sẽ bị prune. [arXiv](https://arxiv.org/html/2404.06109v1)

Đây là chỗ densification “mang mùi heuristic” nhất: nó dùng **magnitude của positional gradient** như một proxy cho “khu vực cần thêm capacity”.

### 2.3 Vì sao heuristic này fail?

Hai failure mode rất nổi tiếng và được phân tích kỹ:

1. **High-frequency nhưng gradient vị trí ≈ 0**
    Revising Densification chỉ ra: vùng có pattern tần số cao, ít Gaussian to cover; di chuyển infinitesimal một Gaussian có thể **không làm error đổi đáng kể**, nên positional gradient nhỏ → không trigger densification, dù error tuyệt đối lớn. [arXiv+1](https://arxiv.org/html/2404.06109v1)
2. **Gradient collision (triệt tiêu hướng)**
    AbsGS phân tích positional gradient thực chất là tổng các sub-gradient theo pixel; pixel gradients có thể ngược hướng nhau nên triệt tiêu → Gaussian lớn (cover nhiều pixel) lại có gradient magnitude nhỏ → không bị split → blur. [arXiv](https://arxiv.org/html/2404.10484v1)

Ngoài ra còn các vấn đề “cơ học” của split/clone gốc:

* EDC chỉ ra clone có thể tạo 2 Gaussian chồng lên nhau và nhận gradient tương tự → khó tách; split sampling ngẫu nhiên và giữ shape con giống cha tạo mismatch hình học → làm chậm tối ưu. [arXiv](https://arxiv.org/html/2411.10133v4)

---

## 3) Các hướng gần đây đang làm gì quanh densification (perspective/angle + problem họ giải)

Mình gom thành các “angle” lớn; mỗi angle là một kiểu tư duy khác nhau.

### Angle A — Sửa “tiêu chí chọn ai để densify” (selection criterion)

**A1) Error-based densification (pixel error)**

* _Revising Densification in Gaussian Splatting_ thay positional-gradient threshold bằng **per-pixel error** (vd SSIM), sau đó phân bổ error về từng Gaussian theo mức đóng góp vào pixel (tỉ lệ với alpha-blending weight), để có “per-Gaussian error score”. [arXiv+1](https://arxiv.org/html/2404.06109v1)
* Họ còn:
    * sửa **opacity bias sau cloning** (giữ opacity y nguyên sau clone làm tăng ảnh hưởng của primitive mới do alpha-compositing), đề xuất giảm opacity sau clone theo nghiệm của một phương trình (trong setting đơn giản). [arXiv](https://arxiv.org/html/2404.06109v1)
    * thêm cơ chế **budget control** để giới hạn số primitives và tránh OOM. [arXiv](https://arxiv.org/html/2404.06109v1)
    * đề xuất “alternative to opacity reset” mượt hơn (giảm opacity đều, kèm regularize residual transmittance). [arXiv](https://arxiv.org/html/2404.06109v1)

**Angle** của họ: densification phải “nhìn vào error tuyệt đối”, chứ không nhìn proxy gradient.

---

**A2) Pixel-aware gradient (sửa cách tính gradient score)**

* _Pixel-GS_ nói positional-gradient average qua views bị lệch vì Gaussian lớn nhìn thấy ở nhiều view nhưng nhiều view chỉ cover ở rìa; họ dùng **số pixel covered** làm trọng số để average gradient, và scale gradient theo khoảng cách camera để giảm floaters. [arXiv+1](https://arxiv.org/abs/2403.15530)

**Angle**: vẫn dùng gradient (để giữ tính “native” của pipeline), nhưng làm gradient score phản ánh coverage.

---

**A3) Gradient-direction aware / “gradient collision”**

* _AbsGS_ đưa khái niệm gradient collision và thay criterion bằng “homodirectional gradient” (tổng trị tuyệt đối sub-gradients theo pixel) để tránh triệt tiêu hướng. [arXiv+1](https://arxiv.org/html/2404.10484v1)
* Các bản sau còn có hướng “gradient coherence ratio” để phân biệt vùng gradient đồng hướng vs xung đột hướng, nhằm split/clone hợp lý hơn. [arXiv+1](https://arxiv.org/html/2508.09239v1)

**Angle**: không chỉ magnitude, mà hình học hướng của gradient cũng quan trọng.

---

### Angle B — Sửa “operation” densify (split/clone làm gì)

**B1) Sửa split/clone để ít phá tối ưu hơn**

* _Efficient Density Control (EDC)_ phân tích rõ nhược điểm của clone/split gốc, rồi đề xuất **Long-Axis Split**: tách theo trục dài nhất, chỉnh vị trí + shape + opacity con để hình dạng vùng phủ trước/sau split gần nhau, giảm overlap; đồng thời đề xuất **Recovery-Aware Pruning** để loại “overfitted Gaussians” dựa trên tốc độ hồi phục opacity sau reset. [arXiv](https://arxiv.org/html/2411.10133v4)

**Angle**: densification là can thiệp “phẫu thuật” vào representation; phải làm sao đừng làm SGD “choáng”.

---

**B2) “Residual split” + supervision đa tỉ lệ**

* _ResGS_ nói densification gốc thiếu adaptiveness và có “dilemma” giữa **coverage hình học** và **recover detail**; đề xuất “residual split” (thêm Gaussian down-scaled như residual), kèm **Gaussian image pyramid** để supervise coarse-to-fine và một selection scheme ưu tiên densify coarse Gaussians theo thời gian. [CVF Open Access+1](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_ResGS_Residual_Densification_of_3D_Gaussian_for_Efficient_Detail_Recovery_ICCV_2025_paper.pdf)

**Angle**: densification nên gắn với multi-scale training (giống ý tưởng multires trong NeRF) để tách “hình” và “vân”.

---

**B3) Phân rã vai trò clone vs split, rồi lên lịch global→local**

* _Decomposing Densification…_ phân tích split/clone có vai trò khác nhau và đề xuất **global-to-local densification**, kèm training framework coarse-to-fine theo energy density và dynamic pruning để tăng tốc. [arXiv](https://arxiv.org/abs/2507.20239)

**Angle**: densification không chỉ là “thêm điểm”, mà là chiến lược phân bổ capacity theo tiến trình tối ưu.

---

### Angle C — “Bẻ lái” sang xác suất / sampling: bỏ heuristic bằng lý thuyết stochastic

**3DGS as Markov Chain Monte Carlo (MCMC)**

* NeurIPS 2024: coi tập Gaussians là mẫu ngẫu nhiên từ một phân phối nền; đưa noise để biến update thành **SGLD**, và “rewrite densification/pruning” như **state transition** của MCMC samples, loại bỏ heuristics; đổi clone thành relocalization gần bảo toàn sample probability; thêm **L1 regularizer** để dùng Gaussians hiệu quả và kiểm soát số lượng. [NeurIPS](https://neurips.cc/virtual/2024/poster/94984)

**Angle**: densification/pruning nên được nhìn như _resampling_ trong sampling/inference, không phải rule-based.

---

### Angle D — Densification khi initialization kém / thiếu cấu trúc

Đây là problem setting cực thực tế: SfM point cloud thiếu → vùng trống / textureless / occlusion.

* Ngay trong related work của EDC có nhắc: **GaussianPro** dùng depth/normal maps để guide densification và lấp gap từ SfM. [arXiv](https://arxiv.org/html/2411.10133v4)
* Có các hướng urban/dynamic: **Visibility-Aware Densification…** (VAD-GS) nhắm tới urban scenes khi frustum overlap không đủ; họ dùng visibility reasoning + view selection + patch matching MVS để sinh Gaussians mới ở vùng thiếu. [arXiv](https://arxiv.org/abs/2510.09364)

**Angle**: ADC kiểu clone/split “chỉ nhân bản cái đang có” → không thể sinh geometry mới nếu ban đầu không có; cần priors hình học.

---

### Angle E — Pruning/compactness (đi kèm densification vì “thêm rồi phải bớt”)

* _PUP 3D-GS (CVPR 2025)_: pruning dựa trên **sensitivity score** từ Fisher Information Matrix (FIM) để giữ Gaussians quan trọng, nhằm nén/giảm size mà giữ chất lượng; họ mô tả cách ước lượng FIM/sensitivity. [arXiv](https://arxiv.org/html/2404.06109v1)
* Các paper kiểu compression/compactness cũng là “mặt âm” của densification: nếu densify không có nguyên lý, pruning phải gánh hậu quả.

---

## 4) Gap chung hiện tại của hướng densification 3DGS

Nhìn tổng thể, cộng đồng đang “đắp vá” ba thứ: **criterion**, **operation**, **schedule/regularization**. Nhưng gap sâu hơn (và là chỗ bạn có thể cắm một theory-based contribution) là:

### Gap 1 — Densification là bài toán _model selection_ nhưng đang làm bằng heuristic

Densification/pruning thực chất là quyết định **thêm/bớt basis functions** (splats). Đây là tối ưu hóa trên không gian mô hình biến thiên (variable complexity). Cách hiện tại: thresholding dựa trên gradient hoặc error proxy.
=> thiếu một nguyên lý kiểu: “tôi đang tối ưu functional nào với regularizer nào, densification là bước gì trong thuật toán đó?”

SRM nói đúng điều này: “Fitting splat models reduces to optimization over the space of mixing measures… WFR gradient flows…” và “recover Gaussian Splatting… disambiguates inverse problem/model/optimization algorithm”.

### Gap 2 — Không có “first-order optimality condition” cho việc _sinh splat mới_

Các paper như Revising/Pixel/AbsGS cải thiện “ai được split/clone”, nhưng vẫn dựa vào “biến thể của densify existing Gaussians”.
Thiếu một cơ chế kiểu: “nếu tôi được phép thêm 1 Gaussian mới bất kỳ, Gaussian tối ưu tiếp theo nên đặt ở đâu/shape nào để giảm loss nhiều nhất?”
Đây là chỗ theory (functional gradient / Frank–Wolfe / boosting / optimal transport) có thể vào.

### Gap 3 — Clone/split can thiệp representation nhưng không được ràng buộc bởi bảo toàn (conservation)

EDC nói split sampling ngẫu nhiên làm “shape distribution trước/sau” lệch nhiều. [arXiv](https://arxiv.org/html/2411.10133v4)
Ở góc measure/transport: split/clone nên gần giống một phép **refinement** giữ gần invariant một số đại lượng (mass/coverage). Nếu làm theo geometry (W2/Bures), bạn có thể thiết kế operation có tính “đẳng cấu” tốt hơn.

### Gap 4 — Thiếu thống nhất giữa densification và pruning dưới một objective/budget

Revising làm budget control; PUP làm pruning theo uncertainty; MCMC thêm L1. [arXiv+2arXiv+2](https://arxiv.org/html/2404.06109v1)
Nhưng nhìn chung vẫn “thêm bằng rule A, bớt bằng rule B”. Một khung theory-based nên hợp nhất thành một bài toán:

$$\min_{\mu} \; \mathcal{L}(\mu) + \lambda \,\mathcal{R}(\mu)$$

trong đó $\mathcal{R}$ điều khiển complexity (số splats / tổng mass / sparsity), và densification/pruning là hệ quả của gradient flow / proximal step.

---

## 5) Một hướng **theory-based** có thể “thay thế ADC densification” nhưng vẫn đúng vibe cộng đồng

Mình đề xuất một “trục chính” (flagship idea) rồi từ đó tách nhánh thành các đề tài publish được.

### Flagship: **WFR Birth–Death Density Control (WFR-BD)**

Ý tưởng: thay vì densify bằng threshold + clone/split, ta xem tập Gaussians là một **measure** $\mu$ trên không gian tham số splat, và tối ưu trực tiếp bằng **Wasserstein–Fisher–Rao gradient flow** như SRM mô tả.

#### 5.1 Form hóa (liên hệ thẳng với SRM)

SRM coi splat regression model như:

$$f_\mu(x)=\mathbb{E}_{(v,A,b)\sim \mu}[v\,\rho_{A,b}(x)] $$

và gradient WFR có dạng (Theorem 1): Fisher–Rao part là kỳ vọng của $\langle \delta F(x),v\rangle \rho_{A,b}(x)$ trừ đi mean; Wasserstein part cho update $v,A,b$.

Trong NVS, forward operator là RTE và thực tế ta render bằng α-blending approximation; $\delta F$ được lấy từ autodiff/backprop qua renderer (SRM nêu rõ).

#### 5.2 “Birth” và “Death” xuất hiện tự nhiên từ Fisher–Rao

* Fisher–Rao geometry cho phép “teleport mass”: tăng/giảm density pointwise (trong space measure).
* Discretize bằng particles: mỗi Gaussian có weight/mass $w_i$ (liên hệ opacity/alpha).
    * nếu FR-gradient gợi ý “mass ở region này nên tăng” → **birth**: thêm particle mới vào vùng đó
    * nếu mass nên giảm → **death**: prune

Điều này thay ADC: thay vì “nếu gradient vị trí lớn thì split/clone”, ta dùng **functional signal** gắn trực tiếp với $\delta F$ (gradient của loss theo field), tức là bám sát mục tiêu tối ưu hóa.

#### 5.3 “Where to spawn?” = bài toán _linear minimization oracle_ kiểu Frank–Wolfe / boos

Splat Regression Models

ng để A*:

Xét bước thêm một splat mới $\theta=(v,A,b)$ với khối lượng nhỏ $\epsilon$. Linearize objective:

$$F(\mu + \epsilon \delta_\theta) \approx F(\mu) + \epsilon \,\langle \nabla_\mu F(\mu), \delta_\theta \rangle$$

Vậy spawn splat mới bằng cách chọn $\theta^*$ sao cho giảm nhiều nhất:

$$\theta^* \in \arg\min_\theta \; \langle \nabla_\mu F(\mu), \delta_\theta \rangle$$

Trong ngôn ngữ SRM, cái inner product này chính là dạng tích phân của $\langle \delta F(x), v\rangle \rho_{A,b}(x)$ (xuất hiện trong Theorem 1).

Thực thi: bạn không cần solve global continuous optimization; bạn có thể làm **candidate sampling + score** (giống MCMC nhưng deterministically), hoặc dùng backprojection residual field (phần 6 bên dưới).

**Điểm ăn tiền**: đây là một cơ chế densification có **nguyên lý tối ưu hóa rõ ràng**, khác hẳn threshold heuristics.

---

## 6) Các hướng nghiên cứu “đủ novel publish A*” (cực cụ thể, có math + setup)

Mình sẽ liệt kê theo cụm. Bạn có thể chọn 1 flagship + 1–2 ablation/extension để thành paper.

### Hướng 1 — **Densification-as-Frank–Wolfe/Boosting trong không gian measure**

**Đặt problem**: NVS/3DGS training, bạn muốn **tăng chất lượng với ít Gaussians hơn** hoặc **đạt target quality với budget cố định**.

Splat Regression Models

n densify không phải split/clone, mà là “thêm basis function tốt nhất” theo linearized objective (functional gradient).

**Math lõi**:

* define objective: $\mathcal{L}(\mu)=\sum_{c\in \text{views}}\ell(R(\mu;c),I_c)$
* compute $\delta \mathcal{L}/\delta f$ qua renderer (autodiff)
* solve: $$\theta^*=\arg\max_\theta \left|\int \langle \delta\mathcal{L}(x), v\rangle \rho_{A,b}(x)\,dx \right|$$ (sign quyết định birth hay kill; hoặc restrict $v$ theo color basis)
* Update $\mu \leftarrow (1-\gamma)\mu + \gamma \delta_{\theta^*}$ hoặc $\mu \leftarrow \mu + \epsilon \delta_{\theta^*}$

**Liên hệ cộng đồng**:

* Revising densification cũng dùng pixel error để chọn nơi densify; bạn thay nó bằng **optimal “atom selection”** dựa trên functional gradient thay vì threshold. [arXiv](https://arxiv.org/html/2404.06109v1)
* MCMC 3DGS muốn bỏ heuristic; hướng này là deterministic counterpart có lý thuyết tối ưu. [NeurIPS](https://neurips.cc/virtual/2024/poster/94984)

**Novelty angle cho A***: “Principled density control via measure-space conditional gradient” + chứng minh giảm objective mỗi bước (ít nhất trong setting đơn giản hóa), kèm empirical SOTA-ish.

---

### Hướng 2 — **Adjoint backprojection field cho “birth”: densification như tomography**

**Đặt problem**: vùng thiếu cấu trúc/khởi tạo kém (SfM thiếu điểm, urban dynamic, occlusion).

**Ý tưởng**: xây một trường 3D “residual potential” từ ảnh—giống idea δF trong inverse problems.

SRM viết tổng quát cho inverse problems:

$$\delta F[f](x)=(D_f A)^*A[f](x)-(D_f A)^*[g](x)$$

điều này là “adjoint residual” (giống PDE/inverse problems).

Trong NVS, $A$ là renderer (RTE/α-blending). SRM nói gradient phụ thuộc approximation và compute bằng autodiff.

**Thực thi kiểu CV**:

* compute per-pixel residual $r(u)=\partial \ell/\partial I(u)$
* backproject residual dọc ray vào 3D với một kernel (ví dụ Gaussian theo depth uncertainty), tạo field $G(x)$ ≈ “nếu đặt splat ở x sẽ giảm loss bao nhiêu”
* spawn new Gaussians ở local maxima của $G(x)$; shape $Σ$ chọn theo local Hessian/structure tensor; màu init theo SH fit local.

**Liên hệ cộng đồng**:

* VAD-GS/gaussianPro dùng priors hình học để sinh Gaussians ở vùng thiếu; bạn đưa một cơ chế _adjoint-based_ có lý thuyết inverse problems. [arXiv+1](https://arxiv.org/abs/2510.09364)
* Revising densification redistributes pixel errors về gaussians theo contribution; bạn đi sâu hơn: biến residual thành 3D potential field. [arXiv](https://arxiv.org/html/2404.06109v1)

**Novelty**: densification không còn “clone t

Splat Regression Models

n từ residual field” có liên hệ toán học rõ.

---

### Hướng 3 — **WFR Birth–Death với regularizer kiểm soát complexity (thay opacity reset + pruning rời rạc)**

**Đặt problem**: kiểm soát số Gaussians + ổn định training (không shock do opacity reset), tránh OOM.

Revising densification chỉ ra opacity reset gây shock và đề xuất cách mượt hơn. [arXiv](https://arxiv.org/html/2404.06109v1)
MCMC thêm L1 regularizer để dùng Gaussians hiệu quả và kiểm soát số lượng. [NeurIPS](https://neurips.cc/virtual/2024/poster/94984)

**Ý tưởng**: thay “reset + prune threshold” bằng một bài toán có regularizer rõ:

$$\min_{\mu}\;\mathcal{L}(\mu) + \lambda \|\mu\|_{1} \quad \text{(hoặc TV/mass/entropy)}$$

Trong discretization particle, $\|\mu\|_1$ ~ tổng trọng số/opacity; gradient FR sẽ tự shrink weights → **death** tự nhiên, không cần reset.

**Math**:

* Fisher–Rao gradient trong SRM có dạng “centered” expectation; nó là nền để làm mass update có tính bảo toàn tổng (mean-centering).
* Với L1/TV regularization, bạn có thể dùng proximal step hoặc birth–death dynamics (liên hệ literature birth–death, SRM cũng nhắc).

**Novelty**: “Unified densification+pruning as WFR gradient descent on regularized objective” → thay thế cả ADC loop.

---

### Hướng 4 — **Operation-theory: Split như một phép tối ưu OT/Bures (thay sampling ngẫu nhiên)**

EDC nói split gốc phá hình dạng; họ sửa bằng long-axis split. [arXiv](https://arxiv.org/html/2411.10133v4)

SRM chỉ ra space của splats có hình học W2 rút về **Bures–Wasserstein metric** cho Gaussian family.

**Ý tưởng**: thiết kế split như một bài toán tối ưu nhỏ:

* bạn muốn thay 1 Gaussian bằng 2 Gaussian sao cho:
    * tổng “mass”/coverage được bảo toàn (theo W2/Bures)
    * đồng thời giảm objective trên residual

Form:

$$\min_{\theta_1,\theta_2} \;\; \alpha\, d_{\text{Bures}}(\theta_{\text{parent}}, \tfrac12\theta_1+\tfrac12\theta_2)\;+\; (1-\alpha)\,\Delta \mathcal{L}$$

Bạn có thể cho closed-form gần đúng dùng eigen-decomposition của Σ (giống long-axis split nhưng

Splat Regression Models

).

**Novelty**: “OT-consistent split” (đặc biệt nếu bạn chứng minh split giảm distortion trong W2 sense) → vừa theory vừa practical.

---

### Hướng 5 — **Multi-scale densification có bảo chứng: wavelet/octree nhưng trên measure**

ResGS dùng image pyramid để supervise coarse-to-fine. [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/papers/Lyu_ResGS_Residual_Densification_of_3D_Gaussian_for_Efficient_Detail_Recovery_ICCV_2025_paper.pdf)
Decomposing densification cũng dùng multi-res/energy guided. [arXiv](https://arxiv.org/abs/2507.20239)

**Ý tưởng theory-based**: xây một representation đa tỉ lệ trong measure:

$$\mu = \sum_{l=0}^{L} \mu^{(l)},\quad \text{mỗi level có scale prior khác nhau}$$

Densification là chuyển mass từ coarse level sang fine level (WFR teleport), theo error estimator.

Bạn có thể lấy cảm hứng từ FEM adaptive refinement (a posteriori error estimator), nhưng mapping sang gaussian splats.

**Novelty**: “principled multi-resolution density control with measure coupling constraints”.

---

### Hướng 6 — **Uncertainty-guided birth (kết hợp PUP với densification, không chỉ pruning)**

PUP dùng FIM/sensitivity để prune. [arXiv](https://arxiv.org/html/2404.06109v1)

**Ý tưởng**: dùng uncertainty như “acquisition function” để spawn:

* vùng nào FIM thấp (không chắc) nhưng residual cao → cần thêm primitives
* hoặc “active densification” theo expected information gain.

**Math**: link giữa FIM và local curvature/Hessian của loss wrt parameters; spawn nơi gradient variance cao.

**Novelty**: biến pruning theory thành densification policy (active capacity allocation).

---

### Hướng 7 — **Densification cho missing geometry: từ “clone existing” → “generate new”**

VAD-GS chỉ ra clone/split không thể reconstruct missing structure. [arXiv](https://arxiv.org/abs/2510.09364)

**Ý tưởng**: kết hợp “adjoint residual field” (Hướng 2) với geometry prior:

* visibility reasoning để loại residual do occlusion/invalid gradients
* spawn candidates từ patch matching MVS / depth prior
* sau đó refine bằng WFR/SGD.

**Novelty**: “visibility-consistent birth operator” — paper kiểu CVPR/ECCV rất hợp.

---

## 7) Nếu bạn muốn _novel hơn nữa_: nối SRM với cộng đồng 3DGS như thế nào cho “đúng gu” A*

Một trick mạnh: **đừng pitch như “tôi có lý thuyết đẹp”**, pitch như:

* “Tất cả paper densification gần đây đang fix symptom: error-based criterion (Revising), pixel-aware gradient (Pixel-GS), gradient collision (AbsGS), operation mismatch (EDC), schedule (Decompose/ResGS), heuristic removal (MCMC).
    Mình unify symptom đó bằng một objective/geometry duy nhất: measure optimization trong WFR space.
    Từ đó, mình derive một densification operator có nguyên lý: birth = giải bài toán chọn atom tối ưu theo first variation; death = shrink từ FR dynamics; split = OT-consistent refinement…”

SRM đã cho bạn “ngôn ngữ” này.

---

## 8) Checklist thực nghiệm để paper “đứng” (rất thực dụng)

Để publish A*, ngoài novelty bạn cần “đúng benchmark + đúng ablation”.

**Datasets chuẩn** (cộng đồng densification hay dùng):

* Mip-NeRF 360 / Tanks&Temples / Deep Blending (Revising, AbsGS, Improved ADC, EDC đều dùng). [arXiv+3arXiv+3arXiv+3](https://arxiv.org/html/2404.06109v1)

**Ablations phải có**:

* #Gaussians vs PSNR/SSIM/LPIPS tradeoff (Revising nhấn mạnh budget control; MCMC nhấn mạnh control; PUP nhấn mạnh pruning). [arXiv+2NeurIPS+2](https://arxiv.org/html/2404.06109v1)
* convergence speed (Decompose/Improving ADC nói speedup). [arXiv+1](https://arxiv.org/abs/2507.20239)
* robustness to bad initialization (Pixel-GS/MCMC/VAD-GS đều xoáy). [arXiv+2NeurIPS+2](https://arxiv.org/abs/2403.15530)

**Baselines bắt buộc**:

* 3DGS default ADC (gốc)
* Revising Densification (error-based) [arXiv](https://arxiv.org/html/2404.06109v1)
* Pixel-GS (pixel-aware gradient) [arXiv](https://arxiv.org/abs/2403.15530)
* AbsGS (gradient collision) [arXiv](https://arxiv.org/html/2404.10484v1)
* (tuỳ scope) EDC / ResGS / MCMC / PUP [arXiv+3arXiv+3CVF Open Access+3](https://arxiv.org/html/2411.10133v4)

---

## Kết luận “ngắn mà sắc”

* Hướng densification 3DGS hiện nay đang chạy theo 3 trục: **criterion**, **operation**, **schedule/regularization**. Những paper mạnh đều hoặc (i) phân tích failure mode của heuristic ADC và sửa, hoặc (ii) thay heuristics bằng framework xác suất, hoặc (iii) tối ưu hóa operation/speed. [arXiv+4arXiv+4arXiv+4](https://arxiv.org/html/2404.06109v1)
* **Gap lớn nhất**: chưa có một cơ chế densification xuất phát từ một bài toán tối ưu hóa/geometry thống nhất.
* **SRM** cho đúng công cụ để làm điều đó: coi gaussians như **mixing measure**, dùng **WFR gradient flow**, từ đó densification = **birth–death** theo first variation δF.
* Một hướng theory-based đủ “A*” là: **WFR-BD density control** (birth bằng functional-gradient/Frank–Wolfe, death bằng FR shrink, split OT-consistent). Nó vừa thay ADC, vừa tương thích với vibe của cộng đồng (vì vẫn nằm trong 3DGS training pipeline).

Nếu bạn tiếp tục theo hướng này, mình khuyến nghị chọn 1 flagship (ví dụ **WFR-BD + atom selection**) và đóng gói theo một _problem setting_ cụ thể (vd **budgeted reconstruction**, hoặc **missing-geometry densification**, hoặc **robust-to-initialization**) để thông điệp paper sắc như dao mổ, không bị “too broad”.

* * *

**You:**

Hãy analysis tính novel và khả thi của mỗi phương pháp (hướng đi) bạn đề xuất, với mỗi phương pháp theo bạn dựa trên theorem gì đã có sẵn trong bài báo Splat Regression Models tôi đính kèm và có phát triển dựa trên theory-based hay không, đề xuất của bạn giải quyết gap và dựa trên lí thuyết gì liên quan đến hướng phát triển chung của cộng động và vấn đề chung trong 3dgs hay không, hay chỉ là hallunicated bịa đặt và assumption. Hãy kiểm tra lại theorem và những bài báo lí thuyết liên quan xem đề xuất có chính xác không, hãy chỉ là một bài toán bịa đặt và không có giá trị, hãy xem xung quanh có ai làm chưa hay chỉ là suy đoán. Sau đó dựa trên phân tích của bạn hãy tổng hợp hướng có giá trị nhất mà bạn tin rằng điều đó có ý nghĩa để theo đuổi để thuyết phục reviewer khi viết báo cho hội nghị A*. Nếu tất cả hướng đều đang có vấn đề thì hãy nhận định tại sao. Tập trung vào theorem để hiểu được theorem Splat Regression Models có thể giải quyết được vấn đề gì.

* * *

**ChatGPT:**

Dưới đây là một “bản đồ” khá nghiêm túc (nhưng không tự nghiêm trọng quá mức) về: **(i)** theorem/khung lý thuyết trong _Splat Regression Models_ (SRM) bạn gửi, **(ii)** cộng đồng 3D Gaussian Splatting (3DGS) gần đây đang sửa densification kiểu gì và vì sao, **(iii)** các hướng _theory-based_ có thể **thay thế densification heuristic hiện tại**, kèm **đánh giá độ novel + khả thi + mức “không bịa”** cho từng hướng, và **(iv)** hướng đáng theo đuổi nhất để “thuyết phục reviewer A*”.

---

## 1) SRM thực sự cho ta cái gì (theorem-level) liên quan trực tiếp đến densification?

### 1.1. 3DGS/“splat” được viết lại thành một bài toán tối ưu trên **không gian measure**

SRM định nghĩa một **splat measure** $\mu$ và hàm mô hình:

$$f_\mu(x) := \mathbb{E}_{(v,\rho_{A,b})\sim \mu}\,[\,v\,\rho_{A,b}(x)\,]$$

với $\rho_{A,b}=(A(\cdot)+b)_\#\rho$. Nếu $\mu$ có $k$ điểm hỗ trợ thì chính là **k-splat** (tương ứng “k Gaussian” trong 3DGS).

Splat Regression Models

Điểm quan trọng: **$f_\mu$ tuyến tính theo $\mu$**. Điều này mở cửa cho cả “gradient flow trên measure” lẫn “Frank–Wolfe/conditional gradient trên measure” (mình sẽ dùng chỗ này để đề xuất hướng novel nhất).

---

### 1.2. Theorem 1: gradient “đúng hình học” cho _cả di chuyển_ lẫn _tăng/giảm khối lượng_ (WFR)

**Theorem 1** là xương sống: cho gradient theo **Wasserstein–Fisher–Rao (WFR)**, gồm hai phần:

* **Wasserstein gradient** (dịch chuyển “vị trí & shape” của splat): cập nhật $v,A,b$ theo các tích phân của $\delta F$ với $\rho_{A,b}$.
* **Fisher–Rao gradient** (teleport/scale khối lượng): một **score** dạng kỳ vọng của $\langle \delta F(x), v\rangle$ dưới $\rho_{A,b}$, trừ đi mean dưới $\mu$.

SRM ghi công thức rất rõ:

$$\nabla^{FR}_\mu F(f_\mu)(v,A,b)
= \int \langle \delta F(x), v\rangle \rho_{A,b}(x)\,\pi(dx)
-\mathbb{E}_{(v,A,b)\sim\mu}\Big[\int \langle \delta F(x), v\rangle \rho_{A,b}(x)\,\pi(dx)\Big]$$

và $\nabla_v,\nabla_A,\nabla_b$ tương ứng.

**Tại sao liên quan densification?**

* Fisher–Rao là “mass teleportation”: **tự nhiên hóa pruning** (khối lượng/opacity → 0) và cũng là nền để **birth–death** (tăng/giảm số hạt). SRM còn có sketch discretization Fisher–Rao bằng accept–reject + KDE để duy trì “smoothed particle”.

    Splat Regression Models

* Wasserstein phần “move” tương ứng update vị trí/covariance — chính là phần mà 3DGS vẫn làm bằng gradient.

SRM còn nói thẳng: từ framework này có thể “recover heuristic methods used to optimize Gaussian Splat models … as regularized risk minimization via WFR gradient descent”.

Splat Regression Models


=> Nghĩa là: **densification hiện tại = heuristic discretization** của một bài toán hình học đúng hơn.

---

### 1.3. Proposition 1: manifold Gaussian có metric W2 = Bures–Wasserstein

SRM chứng minh family splat $\mathrm{BW}_\rho(\mathbb{R}^d)$ là **geodesically convex subset** của $W_2$ và **W2 giữa hai Gaussian-splat** trở thành **Bures–Wasserstein metric** (dạng Frobenius + nuclear norm).

**Ý nghĩa densification/split/merge:**
Nếu bạn split/merge Gaussians mà không tôn trọng geometry này, bạn rất dễ tạo **overlap, redundancy** (đúng cái mà nhiều paper đang chữa bằng heuristic).

---

### 1.4. Theorem 3: bound “bao nhiêu splat đủ để xấp xỉ”

SRM có theorem định lượng: để xấp xỉ Lipschitz $f$ trên miền “nice”, tồn tại k-splat với $k \lesssim \varepsilon^{-2(d+2)}$.

**Ý nghĩa densification:**

* Có “đường biên” giữa **capacity vs error**.
* Nhưng theorem này _không tự động_ cho ta “khi nào split” trong NVS, vì NVS không phải regression đơn giản và $d$ (không gian tham số/không gian cảnh) phức tạp hơn.

---

## 2) Cộng đồng 3DGS gần đây đang làm gì quanh densification? (perspective/angle)

Mình gom các paper theo “góc nhìn”:

### 2.1. Sửa **tiêu chí densification** (heuristic → ít heuristic hơn, nhưng vẫn heuristic)

* **Revisiting Densification / Revising Densification**: tập trung phân tích lại densification trong 3DGS, chỉ ra vấn đề và đề xuất tiêu chí/chiến lược cải thiện. [arXiv](https://arxiv.org/abs/2404.06109)
* **Pixel-GS**: nhấn vào _pixel-aware / gradient-aware_ để densify nhanh và chính xác hơn. [arXiv](https://arxiv.org/abs/2403.15530)
* **AbsGS**: tập trung vào hiện tượng “gradient collision” làm mất chi tiết, đề xuất cơ chế để recover fine details. [arXiv](https://arxiv.org/abs/2404.10484)
* **Efficient Density Control (EDC)**: nhìn thẳng vào vấn đề overlap/redundancy do split/clone; đề xuất density control hiệu quả hơn (ví dụ split theo trục dài, giảm overlap). [arXiv](https://arxiv.org/abs/2411.10133)

=> Angle chung: **giữ pipeline ADC (Adaptive Density Control) nhưng thay trigger/score**.

---

### 2.2. Thêm “prior” về **cấu trúc thị giác / multi-scale**

* **CDC-GS** (Reframing densification…): densification nên tương thích “complexity” (độ phức tạp thị giác), dùng tín hiệu kiểu wavelet/đa tỉ lệ để cân bằng density. [NeurIPS](https://neurips.cc/virtual/2025/loc/san-diego/session/128336)

=> Angle: “densify theo high-frequency / texture complexity”, gần với tư duy SRM “mother splat ~ mother wavelet”, nhưng paper này đã đi theo hướng đó.

---

### 2.3. Densification như một **lịch trình/chiến lược tối ưu hóa**

* **Decomposing Densification**: tách densification thành global-to-local/đa pha để ổn định huấn luyện và cải thiện chất lượng. [arXiv](https://arxiv.org/abs/2507.20239)

=> Angle: “densification là optimization schedule”, không hẳn là score cục bộ.

---

### 2.4. Densification/pruning theo **uncertainty/sensitivity**

* **PUP 3D-GS** (Principled Uncertainty Pruning): pruning dựa trên đo lường “độ quan trọng/độ nhạy” thay vì threshold opacity thô. [arXiv](https://arxiv.org/abs/2406.10219)

=> Angle: “prune có lý do”, nhưng thường **chưa thay hoàn toàn densification**, chủ yếu prune.

---

### 2.5. Densification như **sampling / probabilistic inference**

* **3D Gaussian Splatting as Markov Chain Monte Carlo**: nhìn 3DGS như MCMC để “reconstruct” 3D, dùng sampling perspective. [arXiv](https://arxiv.org/abs/2404.09591)
* **Metropolis-Hastings Sampling for 3D Gaussian Reconstruction**: tương tự hướng sampling, dùng MH. [NeurIPS+1](https://neurips.cc/virtual/2025/poster/119401)

=> Angle: thay densification heuristic bằng **move proposals + accept/reject**.

---

### 2.6. Densification để chữa **missing geometry / occlusion**

* **Visibility-Aware Densification (VAD-GS)**: densify có awareness về visibility/occlusion, nhắm các cảnh urban khó. [arXiv](https://arxiv.org/abs/2510.09364)

=> Angle: densify phải tôn trọng visibility; nếu không sẽ tạo “floaters”/artifact.

---

## 3) “Gap” chung hiện tại quanh densification 3DGS

Sau khi nhìn các hướng trên, gap chung (và reviewer hay chọc) thường là:

1. **Không có một nguyên lý tối ưu thống nhất** cho _split/clone/prune_: đa số vẫn là rule-based + threshold tuning (dù tinh vi hơn).
2. **Densification bị “local trap”**: clone/split từ splat đang có → khó sinh splat vào vùng trống nhưng cần (missing geometry).
3. **Redundancy/overlap**: densify nhiều dẫn đến chồng lấp và tốn compute; EDC chữa phần nào nhưng vẫn heuristic. [arXiv](https://arxiv.org/abs/2411.10133)
4. **Thiếu certificate/tiêu chuẩn dừng “đúng nghĩa”**: tăng k đến khi nào? (Theorem 3 của SRM gợi ý trade-off, nhưng cộng đồng chưa “khóa” nó thành rule).
5. **Tách rời split với merge/prune**: pipeline hay chỉ “add” + “opacity prune”, hiếm khi có merge đúng nghĩa theo geometry.
6. **Thiếu liên kết giữa geometry của Gaussian (Bures/W2)** với thao tác split/merge: nhiều split theo eigenvector là hợp lý, nhưng chưa “hệ quy chiếu đúng” (Proposition 1 chỉ ra hệ quy chiếu)

    Splat Regression Models


SRM cung cấp “cái khung” để biến gap #1–#3 thành một bài toán hình học/measure thống nhất. Vấn đề là: **làm sao discretize ra thuật toán vừa chạy được trên GPU vừa đủ mới để publish**.

---

## 4) Các hướng theory-based có thể “thay thế” densification heuristic hiện tại — và kiểm tra xem có bịa không

Mình đề xuất 6

Splat Regression Models

* **Cơ sở theorem trong SRM** (chỉ rõ Theorem/Proposition nào)
* **Cơ sở lý thuyết ngoài SRM** (paper nền)
* **Có ai làm chưa?** (dựa trên các paper mình tìm thấy)
* **Novelty** / **Khả thi**
* **Rủi ro & assumption** (để bạn biết reviewer sẽ bắn gì)

---

# Hướng 1 — WFR Birth–Death Densification/Pruning (thay ADC bằng “Fisher–Rao resampling”)

### Ý tưởng

Thay vì: “Gaussian nào gradient lớn thì split; opacity nhỏ thì prune”
Ta dùng đúng **Fisher–Rao gradient** trong Theorem 1 để tạo **score** cho mỗi splat, rồi làm **birth–death/resampling**:

* Mỗi splat $i$ có score

$$s_i := \int \langle \delta F(x), v_i\rangle \rho_i(x)\,\pi(dx)$$

* Fisher–Rao gradient là $s_i - \bar s$ (trừ mean).
* Diễn giải: splat nào “có ích” (làm giảm loss theo first-order) thì được tăng mass/nhân bản; splat “có hại/không ích” thì bị kill (mass→0).

Bạn có thể triển khai 2 kiểu:

1. **Replicator / multiplicative weights update** cho opacity/weight
2. **Birth–death particle system**: copy/delete splat theo xác suất tỉ lệ score

SRM còn có sketch discretization Fisher–Rao bằng accept–reject + KDE “smoothed particle”.
=> Đây là “móc” để biến theorem thành thuật toán.

### Theorem SRM dựa vào

* Theorem 1: công thức $\nabla^{FR}$ chính là “prune/clone score” đúng hình học.
* SRM cũng nhấn: heuristic 3DGS có thể xem như regularized risk minimization via WFR GD.

### Lý thuy

Splat Regression Models

một dạng **unbalanced optimal transport**/Wasserstein–Fisher–Rao; có nền tảng trong literature unbalanced OT (ví dụ Liero–Mielke–Savaré về cấu trúc metric/gradient flows; và Chizat et al. về thuật toán cho unbalanced OT). [arXiv+1](https://arxiv.org/abs/1508.07941?utm_source=chatgpt.com)

* Birth–death dynamics: có

    Splat Regression Models

    ath cho sampling/động lực học (Lu–Slepčev–Wang). [ResearchGate](https://www.researchgate.net/journal/Nonlinearity-1361-6544/publication/374225623_Birth-death_dynamics_for_sampling_global_convergence_approximations_and_their_asymptotics/links/6569f77cb1398a779dcea3c4/Birth-death-dynamics-for-sampling-global-convergence-approximations-and-their-asymptotics.pdf?origin=scientificContributions&utm_source=chatgpt.com)

### Có ai làm “giống vậy” trong 3DGS chưa?

* Có các hướng **sampling-based**: MCMC view (Kheradmand et al.) và MH sampling. [arXiv+2arXiv+2](https://arxiv.org/abs/2404.09591)
* Nhưng chúng là **MCMC** (proposal + accept/reject để sampling), không phải **WFR gradient flow / Fisher–Rao resampling** “theo đúng score từ first variation” như SRM viết.
    => Ý tưởng “resampling theo Fisher–Rao score” **chưa thấy xuất hiện trực diện** trong các paper 3DGS mình lướt được (cẩn thận: mình không thể đảm bảo 100% “chưa ai làm” trong toàn bộ arXiv, nhưng trong các nhánh nổi bật thì chưa thấy).

### Novelty

* **Trung bình–cao**: vì community đã có “sampling lens”, nhưng WFR/Fisher–Rao lens là một nhánh khác, có thể bán như “từ heuristic → gradient flow đúng hình học”.

### Khả thi

* **Trung bình**: cần tính $s_i$ hiệu quả. Nhưng thực tế $s_i$ có thể được ước lượng bằng:
    * tích phân Monte Carlo trên các sample $x$ mà splat $i$ cover (pixel footprint),
    * hoặc dùng chính gradient sẵn có từ backprop (3DGS vốn backprop được qua rasterization).

### Rủi ro / assumption (reviewer sẽ hỏi)

* 3DGS renderer là nonconvex, nên “theory guarantee” về hội tụ toàn cục **không có**. Bạn chỉ có thể claim: “đây là discretization của WFR gradient flow, đảm bảo descent cục bộ theo metric WFR” (đúng về mặt hình học), và show empirical + stability.
* Birth–death dễ “population collapse” nếu score noisy → cần variance reduction / tempering.

**Kết luận hướng 1:** không bịa; bám Theorem 1 rất sát; novelty ổn; nhưng phải cạnh tranh narrative với MCMC/MH (phải làm rõ khác biệt).

---

# Hướng 2 — Conditional Gradient / Sliding Frank–Wolfe Densification (đây là hướng mình thấy “đáng tiền” nhất)

### Ý tưởng

Thay vì “split từ splat đang có”, ta xem densification là bài toán **chọn atom mới** trong không gian measure:

Tại trạng thái hiện tại $\mu$, tính first variation $\delta F[f_\mu](x)$.
Ta chọn một splat mới $\theta=(v,A,b)$ sao cho nó “ăn” mạnh nhất vào gradient:

$$\theta^\* \approx \arg\min_{\theta} \;\; \left\langle \delta F(\cdot), \; v\,\rho_{A,b}(\cdot)\right\rangle_{L_2(\pi)}
\;\;\; \Leftrightarrow\;\;\;
\arg\min_{\theta} \int \langle \delta F(x), v\rangle \rho_{A,b}(x)\pi(dx)$$

Bạn sẽ nhận ra integrand này chính là “s_i” trong Theorem 1.

Sau đó:

* Add atom mới vào $\mu$ (densify)
* “Slide” (tối ưu lại) tất cả atoms bằng gradient descent (Wasserstein part / usual backprop).

Đây chính là cấu trúc của các thuật toán **conditional gradient trên measure** kiểu:

* **ADCG (Alternating Descent Conditional Gradient)** cho sparse inverse problems
* **Sliding Frank–Wolfe** cho sparse deconvolution / BLASSO

### Theorem SRM dựa vào

* Definition 2 (first variation): cho phép viết objective linear hóa theo $\delta F$.
* Theorem 1: công thức $ \int \langle\delta F(x),v\rangle\rho_{A,b}(x)$ là “đúng form” để làm LMO (linear minimization oracle) trong Frank–Wolfe trên measure.
* Theorem 3: backing “số atom cần thiết” (capacity–error trade-off) để làm dừng densification có lý.

### Lý thuyết ngoài SRM

* **ADCG** (conditional gradient cho sparse inverse problems): cung cấp framework “thêm atom + descent nội bộ”. [Cambridge University Press & Assessment+1](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/semidiscrete-unbalanced-optimal-transport-and-quantization/766006E30D4E7CF50BE31163E5298B13)
* **Sliding Frank–Wolfe**: phân tích/thiết kế FW khi atom có tham số liên tục (“slide” vị trí). [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
* Các nền tảng Frank–Wolfe/BLASSO trên measure (đảm bảo hội tụ trong setting convex

    Splat Regression Models

    ù hợp) là “theory spine” bạn có thể dựa vào để viết phần lý thuyết mà reviewer A* rất thích.

### Có ai làm kiểu này trong 3DGS chưa?

Mình đã tìm các cụm như “Frank–Wolfe Gaussian Splatting”, “conditional gradient Gaussian Splatting”… không thấy nhánh chính thống. [arXiv+1](https://arxiv.org/abs/2511.14042)

Splat Regression Models

on hiện tại vẫn chủ yếu:

* sửa heuristic score (AbsGS/Pixel-GS/EDC/CDC-GS/Decomposing…) [NeurIPS+4arXiv+4arXiv+4](https://arxiv.org/abs/2404.10484)
* hoặc sampling MCMC/MH [arXiv+1](https://arxiv.org/abs/2404.09591)

=> Conditional gradient densification là “làn gió lạ” nhưng **không trái cộng đồng** (vì vẫn add Gaussians, chỉ là add theo nguyên lý tối ưu thay vì threshold).

### Novelty

* **Cao** trong hệ 3DGS densification (theo những gì mình thấy từ literature hiện tại).
* Nó biến densification thành một thuật toán có “xương sống” optimization theory (FW/ADCG), thay vì engineering tweak.

### Khả thi

* **Trung bình** (không phải “khó quá không làm được”, mà là “phải thiết kế LMO thông minh”).

LMO “chuẩn” là tối ưu liên tục trên $(v,A,b)$ — khó. Nhưng bạn có thể làm **xấp xỉ có kiểm soát**:

* **Candidate set**: sinh một tập ứng viên $b$ từ backprojection của residual/pixel error (giống tinh thần VAD-GS nhưng không cần MVS), rồi chọn candidate tốt nhất theo score $\int \langle\delta F,v\rangle\rho$.
* Với $A$: chọn isotropic nhỏ hoặc theo local Hessian/gradient direction; refine sau bằng “slide” step.
* Với $v$: lấy từ gradient đóng vai “best response” (ví dụ $v \propto -\int \delta F(x)\rho(x)dx$).

### Rủi ro / assumption

* Lý thuyết FW/BLASSO thường cần **convexity**. Renderer 3DGS (alpha compositing) không convex.
    → Cách viết paper khôn ngoan:
    1. chứng minh/nhắc guarantee trong **setting tuyến tính hóa** hoặc một “proxy problem” convex (để reviewer thấy bạn không bịa),
    2. rồi nói: thực tế NVS nonconvex nhưng thuật toán vẫn là “principled greedy descent” dựa trên first variation.

**Kết luận hướng 2:** rất ít “bịa”; có theorem SRM + backbone optimization literature rất thật; novelty cao; khả thi nếu bạn thiết kế LMO xấp xỉ tốt.

---

# Hướng 3 — Bures–Wasserstein Consistent Split/Merge (thay split heuristic bằng geometry đúng)

### Ý tưởng

Các split kiểu “long-axis split” (EDC) rất hợp lý, nhưng vẫn heuristic.
SRM nói rõ: Gaussian family có geometry W2 = **Bures–Wasserstein**.

=> Ta thiết kế thao tác split/merge như bài toán tối ưu trên manifold đó:

* **Split** một Gaussian $(b,\Sigma)$ thành hai Gaussian $(b_1,\Sigma_1)$, $(b_2,\Sigma_2)$ sao cho:
    * (a) giảm loss theo first variation (theo gradient), và
    * (b) giữ “distortion” nhỏ theo Bures/W2 (tức split không làm “đổ vỡ” hình học).
* **Merge** hai Gaussian gần nhau (và redundant) bằng barycenter trong Bures metric.

### Theorem SRM dựa vào

* Proposition 1: công thức W2/Bures giữa splats (liên quan $A$, nuclear norm) và geodesic convexity.

### Lý thuyết ngoài SRM

* Bures–Wasserstein geometry giữa SPD matrices có literature mạnh (Modin; Bhatia–Jain–Lim). [Optica Publishing Group+1](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-11-18128)

### Có ai làm chưa?

* **EDC** có split theo trục dài để giảm overlap (rất gần tinh thần “respect covariance geometry”, nhưng không gọi tên Bures/W2). [arXiv](https://arxiv.org/abs/2411.10133)
* Có cả nhánh dùng OT để “structuring/compaction” Gaussian splatting (ví dụ Gaussian Herding / GaussianCube) — nhưng thường nhắm compaction/generative modeling, không phải split/merge theo Bures để thay ADC. [arXiv+1](https://arxiv.org/abs/2506.09534)

=> Hướng này **khá có đất**, nhưng novelty không “đột biến” như Hướng 2.

### Novelty

* **Trung bình**: vì cộng đồng đã bàn overlap và split theo covariance. Nhưng nếu bạn làm **merge + split có chứng minh “optimal in Bures sense”** thì vẫn có thể publish tốt.

### Khả thi

* **Cao–trung bình**: Bures computations cần matrix square root/SVD, nhưng d=3 (3D covariance) nên rất nhẹ.

### Rủi ro

* Nếu paper chỉ là “đổi công thức split” mà không có “why this should be optimal”, reviewer sẽ bảo incremental so với EDC.
* Cách tránh: gắn nó với SRM Proposition 1 + một objective phụ “minimize Bures distortion” và show ablation mạnh.

**Kết luận hướng 3:** không bịa; dễ làm; nhưng phải cẩn thận để không bị xem là “EDC++”.

---

# Hướng 4 — Adjoint/Residual Backprojection Birth (densify vào vùng trống, giải missing geometry)

### Ý tưởng

SRM Example 2 nhắc inverse problems: với $F(f)=\frac12\|A[f]-g\|^2$, first variation có dạng “adjoint của residual”.

Trong NVS, $A$ là renderer (gần Radiative Transfer Equation), nên $\delta F$ chính là “backprop residual qua renderer”.

=> Ta tạo một “trường” $r(x)$ trong 3D biểu thị: “nếu đặt density/color tại $x$ thì giảm loss mạnh tới mức nào”.
Rồi **sample births** của Gaussian centers $b$ từ trường này (không cần clone từ Gaussian hiện hữu).

Điều này giải gap: densification clone/split truyền thống khó tạo splat trong vùng “trống” nhưng cần cấu trúc.

### Theorem SRM dựa vào

* Definition 2 + Example 2 + Theorem 1 (score dựa trên $\delta F$).

### Lý thuyết ngoài SRM

* Adjoint methods là chuẩn trong inverse problems; SRM đã đưa công thức first variation cho dạng này.

### Có ai làm gần gần?

* VAD-GS nhắm visibility-aware densification, thường dựa các cues/geometry để densify đúng chỗ. [arXiv](https://arxiv.org/abs/2510.09364)
* Một số paper “point cloud densification” cho 3DGS cũng tồn tại. [OpenReview](https://openreview.net/forum?id=vW5070FoXi&noteId=wmfIKrNmtV)

    Splat Regression Models

    => Hướng này **có họ hàng** với VAD/geometry-guided densification, nhưng “adjoint residual field” là framing lý thuyết hơn.

###

Splat Regression Models

h**: vì “densify từ error/backprojection” nghe giống nhiều trick trong NVS. Novelty tăng mạnh nếu bạn:

* làm nó “đúng” theo first variation (theory),
* và chứng minh nó là một bước “greedy descent” giống conditional gradient (liên kết với Hướng 2).

### Khả thi

* **Trung bình**: phải ước lượng trường 3D; nhưng bạn có thể làm bằng sampling theo ray (giống volume rendering).

### Rủi ro

* Backprojection trong NVS có ambiguity (depth uncertainty), dễ sinh floaters.
* Reviewer sẽ hỏi: “khác gì so với thêm point từ photometric error?” → bạn cần math hóa nó.

**Kết luận hướng 4:** không bịa; khả th

Splat Regression Models

i” đúng để không bị xem là heuristic disguised.

---

# Hướng 5 — Multi-scale / Complexity-guided Densification theo Theorem 3 (cẩn thận: đã có người làm)

### Ý tưởng

Dùng Theorem 3 để nói: k-splat có trade-off capacity–error, nên densification nên theo multi-scale error estimator (wavelet/high-freq).

### Vấn đề: novelty thấp

Vì **CDC-GS** đã làm đúng “density vs complexity” theo tín hiệu wavelet/đa tỉ lệ. [NeurIPS](https://neurips.cc/virtual/2025/loc/san-diego/session/128336)

### Kết luận hướng 5

* **Không bịa**, nhưng **khó publish A*** nếu chỉ làm “SRM + CDC-GS kết hôn”.
* Muốn cứu: phải có theorem mới/guarantee mới hoặc unify chặt với WFR/conditional gradient.

---

# Hướng 6 — Uncertainty/Fisher Information Densification (cẩn thận: dễ bị xem incremental)

### Ý tưởng

PUP 3D-GS làm pruning dựa trên “uncertainty/sensitivity”. [arXiv](https://arxiv.org/abs/2406.10219)
Bạn mở rộng: densify tại vùng có **high uncertainty + high residual**, hoặc dùng natural gradient theo information geometry.

### Novelty

* **Thấp–trung bình** (vì pruning theo uncertainty đã có; densify thêm vào dễ bị xem là extension).

### Khả thi

* **Cao** (engineering khá rõ).

### Kết luận hướng 6

Không bịa, nhưng “A* potential” chỉ cao nếu bạn làm được một **lý thuyết mạnh** nối uncertainty với WFR geometry (khó, nhưng không phải bất khả thi).

---

## 5) Tổng hợp: Hướng nào “đáng theo” nhất để thuyết phục reviewer A*?

Nếu bạn muốn một hướng **(i) thật sự theory-based, (ii) thay thế densification heuristic, (iii) đủ novel so với các paper densification hiện tại, (iv) vẫn bám đường cộng đồng**, thì mình đặt cược vào:

## ✅ Hướng đáng giá nhất: **Conditional Gradient / Sliding Frank–Wolfe Densification (Hướng 2)**

kèm (rất khuyến nghị) một module phụ **Bures-consistent split/merge (Hướng 3)** để giải redundancy.

### Vì sao reviewer A* dễ “nuốt” hướng này?

1. **Bạn có theorem SRM làm cầu nối trực tiếp đến densification score** (Theorem 1 + first variation): không phải “mình nghĩ vậy”.
2. Bạn dựa trên một nhánh **optimization theory có thật** (ADCG / Sliding Frank–Wolfe) — reviewer ML/CV rất quen, dễ tin. [EPFL Biomedical Imaging Group+3Cambridge University Press & Assessment+3Nick Boyd+3](https://www.cambridge.org/core/journals/european-journal-of-applied-mathematics/article/semidiscrete-unbalanced-optimal-transport-and-quantization/766006E30D4E7CF50BE31163E5298B13)
3. Nó **giải gap lớn**: “densification chỉ split từ splat đang có” → FW cho phép add atom _ở bất kỳ đâu_ (nếu LMO/candidate generator tốt).
4. Nó **khác hẳn** các paper hiện tại vốn vẫn là heuristic triggers (AbsGS/Pixel-GS/EDC/Decomposing/CDC) hoặc sampling MCMC/MH. [arXiv+6arXiv+6arXiv+6](https://arxiv.org/abs/2404.10484)
5. Bạn có thể viết phần “theory” hợp lệ:
    * chứng minh trong setting “linearized renderer / simplified forward model” thì conditional gradient có guarantee (dựa literature),
    * còn trong full 3DGS thì nó là greedy descent dựa trên first variation — vẫn principled.

### Một “đề cương thuật toán” đủ cụ thể để thay ADC

* **Loop**:
    1. Run gradient descent vài bước (như bình thường).
    2. Tính residual/gradient $\delta F$ (cái này engine đã có).
    3. Sinh candidate centers $b$ bằng backprojection/triangulation/probabilistic depth sampling từ residual.
    4. Với mỗi candidate, ước lượng score $s(b,A,v)=\int\langle\delta F(x),v\rangle \rho_{A,b}(x)\pi(dx)$.
    5. Chọn top-1 hoặc top-M candidate (Frank–Wolfe step), add vào splat set.
    6. Optional: dùng **Bures split/merge** để refine/giảm overlap theo Proposition 1.
    7. Prune: có thể dùng Fisher–Rao score (từ Theorem 1) thay opacity threshold.

### “Điểm đau” bạn cần giải trước khi viết paper

* Thiết kế LMO xấp xỉ sao cho chạy nhanh (candidate generation tốt là 80% thắng lợi).
* Trình bày “theory claim” khi nonconvex: phải rõ bạn chỉ guarantee trong proxy convex, còn full NVS là empirical + geometric motivation.

---

## 6) Nếu bạn muốn một paper “thuần SRM/WFR” hơn (Plan B tốt)

Nếu bạn nghiêng về “geometry/optimal transport” và muốn paper mang mùi “hình học sâu”, thì:

* **WFR Birth–Death Density Control (Hướng 1)** là Plan B rất hợp: nó bám Theorem 1 cực sát, và bạn định vị khác với MCMC/MH bằng “gradient flow vs sampling”.

Nhưng nhớ: community đã có sampling lens, nên novelty phải đến từ **(i)** score đúng theo first variation (SRM), **(ii)** discretization GPU-friendly, **(iii)** kết quả thực nghiệm vượt các baseline MCMC/MH.

---

## Chốt lại (không vòng vo)

* **Theorem quan trọng nhất của SRM cho densification** là **Theorem 1**: nó cho “score” dạng $\int \langle\delta F,v\rangle \rho$ — đây chính là chìa khóa để biến densification từ heuristic thành optimization principle.
* **Hướng novel + có lý thuyết + có khả năng publish A*** nhất theo mình là:
    **Conditional Gradient / Sliding Frank–Wolfe densification** (Hướng 2), kết hợp **Bures-consistent split/merge** (Hướng 3) như một module hình học “đúng metric”.

Nếu bạn theo hướng này, bạn đang nói với reviewer một câu rất “đáng tin”:

> “Chúng tôi không chỉnh threshold; chúng tôi thay thế densification bằng một thuật toán tối ưu hóa trên measure, có backbone theory (FW/ADCG) và score xuất phát từ first variation đúng theo SRM.”

Và đó là thứ reviewer A* thường khó ghét (trừ khi implementation yếu).

Nếu bạn muốn, ở bước tiếp theo mình có thể phác **một section “Method” kiểu paper-ready**: định nghĩa objective, LMO xấp xỉ, thuật toán, complexity, và một “Theory” phần proxy convex + liên hệ trực tiếp Theorem 1/Proposition 1 (để reviewer đọc thấy “đinh”).

* * *

**You:**

Hướng 2: Conditional Gradient / Sliding Frank–Wolfe Densification trông có vẻ khả thi, nhưng bạn chưa formalize và verify nó detail hơn. Là một reviewer, nếu bạn đọc bài báo trình bày ý tưởng như thế, bạn sẽ đánh giá thế nào theo criteria của hội nghị, liệu ý tưởng có đang dựa trên theory-based kết hợp với computer vision trong 3dgs chính xác không hay chỉ là assumption và bạn tin rằng nó đủ tốt. Hãy phân tích kĩ càng để sau này chúng ta không bị stuck và chuyển đổi sang ý tưởng khác, vì từ bước idea đến implementation cần phải thống nhất với nhau. Hãy đảm bảo phân tích chi tiết và đảm bảo chúng ta đang đi đúng hướng. Tốt nhất bạn nên kiểm tra kĩ lại và draft một section “Method” kiểu paper-ready: định nghĩa objective, LMO xấp xỉ, thuật toán, complexity, và một “Theory” phần proxy convex + liên hệ trực tiếp Theorem 1/Proposition 1. Kiểm tra theory formula chi tiết

* * *

**ChatGPT:**

Đặt mũ “reviewer A*” lên đầu, ý tưởng **Conditional Gradient / Sliding Frank–Wolfe Densification** (mình gọi ngắn là _SFW-densify_) chỉ có cơ hội sống sót nếu bạn làm được 3 việc rất “cứng”:

1. **Formalize** densification như một bước tối ưu hóa hợp lý (không phải “clone/split vì thấy gradient to”).
2. **Chứng minh liên hệ trực tiếp** với lý thuyết trong _Splat Regression Models_ (SRM), cụ thể **Definition 2 + Theorem 1 + Proposition 1**, chứ không phải gắn mác theory cho sang.

    Splat Regression Models

    Splat Regression Models

    Splat Regression Models

3. **Chứng minh** ý tưởng này _không phải bạn bịa_ bằng cách đối chiếu với các khung lý thuyết đã có: **Frank–Wolfe / Conditional Gradient**, **ADCG**, **Sliding Frank–Wolfe cho BLASSO**. [EPFL Biomedical Imaging Group+3Wiley Online Library+3Infoscience+3](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800030109)

Mình sẽ đi theo đúng checklist đó: (A) sanity check toán học + không hallucinate, (B) draft “Method” paper-ready, (C) draft “Theory” paper-ready gắn trực tiếp Theorem 1/Prop 1, rồi (D) reviewer đánh giá: novelty–soundness–feasibility.

---

## A. Sanity check: Ý tưởng FW densification có khớp SRM không?

### A1) Điểm móc quan trọng nhất trong SRM: **First variation** và **Theorem 1**

SRM định nghĩa **first variation** của một functional $F: L^2_\pi(\mathbb{R}^d;\mathbb{R}^p)\to\mathbb{R}$ là hàm $\delta F[f](x)$ sao cho, với mọi nhiễu $\xi$,

$$\left.\frac{d}{d\varepsilon}F(f+\varepsilon\xi)\right|_{\varepsilon=0}
=\int_{\mathbb{R}^d}\langle \xi(x),\delta F[f](x)\rangle\,\pi(dx).$$

Và SRM mô t

Splat Regression Models

viết theo đúng notation SRM):

* mỗi atom là một splat density $\rho_{A,b}$ (pushforward của “mother splat” $\rho$) và một vector $v$
* mô hình là kỳ vọng / mixture:

$$f_\mu(x)=\mathbb{E}_{(v,A,b)\sim\mu}[\,v\,\rho_{A,b}(x)\,].$$

**Theorem 1

Splat Regression Models

t theo hình học Wasserstein–Fisher–Rao, và _đặc biệt quan trọng_ với mình: thành phần gradient theo $v$ là

$$\nabla\nabla_v F(f_\mu)(v,A,b)= \int \delta F(x)\,\rho_{A,b}(x)\,\pi(dx).$$

Nói theo ti

Splat Regression Models

ỗi $\delta F$” được **lọc (smoothed)** bởi kernel $\rho_{A,b}$. Đây chính là _điểm nối_ sang Frank–Wolfe.

---

### A2) FW/Conditional gradient densification chính là “thêm atom” bằng **hướng giảm bậc nhất**

Trong Frank–Wolfe, bạn đang tối ưu trên một miền convex (thường là convex hull của atoms). Mỗi bước bạn giải **linear minimization oracle (LMO)**:

$$s_t \in \arg\min_{s\in\mathcal{D}}\ \langle s,\nabla F(f_t)\rangle,$$

rồi cập nhật $f_{t+1}=(1-\gamma)f_t+\gamma s_t$. [Wiley Online Library+1](https://onlinelibrary.wiley.com/doi/abs/10.1002/nav.3800030109)

Trong bài toán dạng hàm, “gradient” chính là $\delta F$ theo SRM Definition 2. Còn “một atom splat” là:

$$s_{\theta}(x) = v\,\rho_{A,b}(x),
\quad \theta=(v,A,b).$$

Directional derivative khi thêm một atom (nhỏ) là:

$$\left.\frac{d}{d\varepsilon}F(f_t+\varepsilon\,v\rho_{A,b})\right|_{\varepsilon=0}
=\int \langle v\rho_{A,b}(x),\delta F[f_t](x)\rangle\,\pi(dx)
=\Big\langle v,\underbrace{\int \delta F(x)\rho_{A,b}(x)\pi(dx)}_{g(A,b)}\Big\rangle.$$

**Và đây là điểm “không hallucinate”:** cái vector $g(A,b)$ trong ngoặc chính là **$\nabla\nabla_v F$** mà SRM Theorem 1 đã cho.

=> **LMO cho việc “sinh Gaussian mới” đúng ra là tìm $(A,b,v)$ sao c

Splat Regression Models

(g(A,b)) giảm nhiều nhất.**

---

### A3) Nếu ràng buộc $v$ (điều bắt buộc để miền convex compact), LMO có nghiệm đóng dạng

Frank–Wolfe chuẩn cần miền $\mathcal{D}$ **compact**. Vì thế bạn phải _ràng buộc_ độ lớn $v$, ví dụ $\|v\|_2\le V$ hoặc $\|v\|_\infty\le V$ (hợp lý vì SH coeffs / màu thường bị clamp). Khi đó:

$$\min_{\|v\|\le V}\langle v,g(A,b)\rangle = -V\|g(A,b)\|_*$$

(với $\|\cdot\|_*$ là dual norm). Nghiệm:

$$v^*(A,b)=-V\,\frac{g(A,b)}{\|g(A,b)\|}.$$

=> LMO giảm còn:

$$(A^*,b^*) \in \arg\max_{A,b}\ \|g(A,b)\|.$$

Điểm này cực quan trọng về “paper soundness”: bạn biến bài toán chọn $v$ thành closed-form, chỉ còn search $(A,b)$.

---

### A4) “Sliding” là gì và nó có precedent lý thuyết?

**ADCG** và **Sliding Frank–Wolfe (SFW)** đều có cùng pattern:

* _Convex step:_ thêm atom mới theo LMO (global greedy)
* _Nonconvex/local step:_ “trượt” (slide) vị trí/shape atom để cải thiện nhanh

ADCG nói rất thẳng: bước local search là nonconvex nhưng giúp hội tụ nhanh, trong khi vẫn giữ được bound của CGM cho bài toán convex gốc. [Nick Boyd](https://nboyd.github.io/adcg.pdf)

SFW cho BLASSO (sparse measure recovery) cũng y chang: thêm spike bằng FW, rồi solve weights, rồi “move positions” bằng tối ưu không lồi; và họ còn có **finite termination** dưới giả thiết “dual certificate nondegenerate”. [EPFL Biomedical Imaging Group](https://bigwww.epfl.ch/publications/denoyelle1901.pdf)

=> Về mặt “theory culture”: _SFW-densify_ là một bước đi rất đúng “community direction” của tối ưu hóa measure-sparsity, không phải tự nghĩ ra.

---

### A5) Proposition 1 của SRM giúp gì cho “sliding” trong 3DGS?

SRM Proposition 1 nói tập các splat densities $\rho_{A,b}$ là một “generalized Bures–Wasserstein manifold” và khoảng cách Wasserstein giữa hai Gaussian-splats rơi về Bures–Wasserstein metric:

$$W_2^2(\rho_{A,b},\rho_{R,s})
= \|b-s\|^2 + \|A\|_F^2 + \|R\|_F^2 -2\|A^TR\|_*.$$

Điều này không chỉ “nice to cite”. Nó cho bạn:

* một **hình học đúng** để “slide” covariance/shape thay vì update Euclidean tuỳ tiện
* một connection sang liter

    Splat Regression Models

    s–Wasserstein (Modin; Bhatia–Jain–Lim; Chewi et al. về gradient methods trên BW manifold). [arXiv+2ScienceDirect+2](https://arxiv.org/abs/1601.01875)

=> Nếu bạn muốn claim “theory-based sliding”, đây là chỗ _cắm cờ_.

---

### A6) Một điểm kỹ thuật cần cực kỳ cẩn thận: dấu trong $\nabla_A$ của SRM

Trong proof appendix, SRM tính:

$$\nabla_A\big(\rho(A^{-1}(x-b))|\det A|^{-1}\big)
= -\rho_{A,b}(x)\Big(I + \nabla_x\log\rho_{A,b}(x)(x-b)^T\Big)A^{-T}.$$

Nhưng statement Theorem 1 trong phần main text cho $\nabla\nabla_A F$ **không có dấu “–”** (ít nhất trong trích đoạn mình thấy).

Reviewer

Splat Regression Models

này. Cách xử lý “đúng bài” trong paper của bạn:

* Bạn **không cần** dựa vào công thức $\nabla_A$ để định nghĩa LMO. LMO dựa vào Def

    Splat Regression Models

    \int\delta F\rho_{A,b}).
* Với “sliding step”, nếu bạn dùng SRM gradient theo $A$ thì bạn nên **tự derive lại** (từ identity ở proof) để fix sign và ghi chú rõ “we follow the derivation in Appendix B.2”.

Điểm này nhỏ nhưng cứu bạn khỏi reviewer #2.

---

## B. “Method” (paper-ready): Conditional Gradient / Sliding Frank–Wolfe Densification cho 3DGS

Mình sẽ viết như một section trong paper (chỉ cần dịch sang English là nộp được). Chỗ nào là approximation mình sẽ nói rõ.

### B1) Scene representation như một bài toán tối ưu hóa trên measure của splats

Gọi $\Theta$ là không gian tham số của một Gaussian splat:

* $b\in\mathbb{R}^3$: center
* $A\in\mathbb{R}^{3\times 3}$: shape matrix, covariance $\Sigma=AA^T$ (SPD)
* $v\in\mathbb{R}^p$: appearance vector (RGB hoặc SH coefficients; có thể bao gồm opacity/amplitude)

Mother splat $\rho$ (ví dụ Gaussian chuẩn), và splat density:

$$\rho_{A,b}(x)=\rho(A^{-1}(x-b))|\det A|^{-1}.$$

Một scene-field (ví dụ emission coefficients trong SRM Example 2, hoặc một “radiance-like field” tương thích renderer) được biểu diễn dưới dạng splat model:

f_\​:contentReference[oaicite:23]{index=23},A,b)\sim\mu}[v\,\rho_{A,b}(x)],

trong đó $\mu$ là một measure trên $\Theta$.

Với $\mu$ rời rạc $k$-atom: $\mu=\sum_{i=1}^k w_i\delta_{(v_i,A_i,b_i)}$, ta có:

$$f_\mu(x)=\sum_{i=1}^k w_i\,v_i\,\rho_{A_i,b_i}(x).$$

(đây chính là “densi

Splat Regression Models

**Ràng buộc để FW hợp lệ (compactness):** đặt $\|v\|\le V$, $A\in\mathcal{A}$ (ví dụ eigenvalues của $\Sigma$ trong $[\sigma_{\min}^2,\sigma_{\max}^2]$), và

Splat Regression Models

athbb{R}^3) (bounding box scene). Khi đó set atoms bị chặn.

### B2) Objective (mô hình hóa theo SRM / 3DGS)

Ta có bộ ảnh quan sát $\{I_c\}_{c=1}^C$ và camera parameters. Gọi $\mathcal{R}$ là differentiable renderer (3DGS rasterization / alpha compositing), tạo ảnh dự đoán:

$$\hat I_c = \mathcal{R}(f_\mu; c).$$

Mục tiêu:

$$\min_{\mu\in\mathcal{M}} \ \mathcal{L}(\mu):=\sum_{c=1}^C \ell(\mathcal{R}(f_\mu;c), I_c) \ +\ \lambda\,\mathcal{R}_{\text{reg}}(\mu).$$

SRM Example 2 thực chất coi NVS là một spatial inverse problem với forward operator là RTE hoặc xấp xỉ $\alpha$-blending, và $\delta F$ được tính qua linearization/adjoint hoặc autodiff.

**Quan trọng:** $\mathcal{L}$ theo $\mu$ là không lồi (nonconvex) vì $\mathcal{R}$ có occlusion/alpha. Nhưng densification-step vẫn có thể định nghĩa theo “bậc

Splat Regression Models

(xem B3).

### B3) Linear Minimization Oracle (LMO) cho “densification”

Ta dùng notion **first variation** $\delta \mathcal{L}[f_\mu](x)$ theo Definition 2 của SRM:

Xét việc thêm một atom $(v,A,b)$ với “biên độ nhỏ” $\varepsilon$:

$$f_{\mu,\varepsilon}(x)= f_\mu(x) + \varepsilon\, v\rho_{A,b}(x).$$

Đạo hàm bậc nhất:

\​:contentReference[oaicite:31]{index=31}ilon}\mathcal{L}(f_{\mu,\varepsilon})\right|_{\varepsilon=0} =\int \langle v\rho_{A,b}(x),\delta \mathcal{L}[f_\mu](x)\rangle\,\pi(dx) =\langle v, g_\mu(A,b)\rangle,

trong đó

$$g_\mu(A,b) := \int \delta \mathcal{L}[f_\mu](x)\,\rho_{A,b}(x)\,\pi(dx).$$

Theo SRM Theorem 1, $g_\mu(A,b)$ chính là $\nabla\nabla_v \mathcal{L}(f_\mu)(v,A,b)$.

**LMO (nguyên bản):**

$$(A^*,b^*,v^*) \in \arg\min_{A\in\mathcal{A},b\in\Omega,\|v\|\le V}\ \langle v,g_\mu(A,b)\rangle.$$

Vì tối ưu theo $v$ là tuyến tính và

Splat Regression Models

V\frac{g_\mu(A,b)}{|g_\mu(A,b)|}.
]
Do đó:

$$(A^*,b^*)\in \arg\max_{A,b}\ \|g_\mu(A,b)\|.$$

**Ý nghĩa CV:** “điểm cần densify” là nơi _một Gaussian patch_ (kernel $\rho_{A,b}$) có tương quan lớn nhất với “error signal” $\delta \mathcal{L}$ trong không gian 3D.

### B4) Xấp xỉ LMO trong 3DGS (chỗ khó nhất nhưng khả thi)

Vấn đề thực dụng: $\delta \mathcal{L}[f_\mu](x)$ là một trường 3D “adjoint” của renderer; bạn hiếm khi có nó dưới dạng analytic. Có 2 route khả thi:

**Route 1: build một 3D error field bằng backprojection/Monte Carlo (giống adjoint tomography).**

* Tính residual per-pixel $r_{c}(u) = \partial \ell(\hat I_c(u),I_c(u))/\partial \hat I_c(u)$.
* Xấp xỉ $\delta \mathcal{L}(x)\approx \sum_{c}\int_{\text{ray}(c,u)} w(t)\,r_c(u)\,dt$ với $w(t)$ phụ thuộc transmittance/visibility (lấy từ renderer hiện tại).
* Lưu $\delta \mathcal{L}$ trên voxel grid / sparse grid, rồi tính $g_\mu(A,b)$ bằng convolution Gaussian (hoặc Monte Carlo sample từ $\rho_{A,b}$ và nội suy).

**Route 2: candidate set + one-step “virtual atom gradient”.**

* Lấy một tập candidate centers $b$ bằng cách backproject các pixel error lớn theo depth hiện tại (depth từ 3DGS render).
* Với mỗi candidate $b$, set $A$ theo rule (isotropic theo depth / spacing), và ước lượng score bằng cách render “một atom” lên vài view lân cận và tính inner product với residual (approximate first-order gain).
* Chọn top-M candidates.

Route 2 thường dễ implement hơn trong codebase 3DGS hiện có (vì bạn đã có rasterizer), dù “theory purity” kém Route 1 một chút. Nếu nộp A*, bạn có thể trình bày Route 2 như **approximate LMO** (giải trên một tập con candidate), và cite Jaggi: FW vẫn giữ rate nếu LMO giải xấp xỉ với sai số $\delta$ (trong bài Jaggi có khung primal–dual gap và inexact oracle). [Infoscience+1](https://infoscience.epfl.ch/server/api/core/bitstreams/eab3824d-34c3-4014-be75-58e3aad8dc35/content)

### B5) Thuật toán tổng thể: SFW-densify (ADCG-style) trong 3DGS

Mình viết theo phong cách ADCG/SFW:

**Algorithm: Sliding Frank–Wolfe Densification for 3DGS**

**Input:** init gaussians từ SfM; budget $K_{\max}$; interval $T$; step sizes.
**State:** tập atoms $\{(w_i,v_i,A_i,b_i)\}_{i=1}^K$.

Lặp iter = 1..N:

1. **Local optimization (standard 3DGS step).**
    Update $(w_i,v_i,A_i,b_i)$ bằng Adam/SGD trên photometric loss (có thể dùng SRM gradients như một “natural geometry” variant).
2. Nếu iter mod $T=0$ và $K<K_{\max}$: **Densification via (approx) LMO**
    * Compute “error signal”: residual maps và/hoặc $\delta \mathcal{L}$ field.
    * (Approx LMO) tìm $(A^*,b^*)\approx\arg\max_{A,b}\|g_\mu(A,b)\|$.
    * Set $v^*=-V\frac{g_\mu(A^*,b^*)}{\|g_\mu(A^*,b^*)\|}$.
    * **Add atom:** $K\leftarrow K+1$, append $(w_{K},v_{K},A_{K},b_{K})=(\gamma, v^*, A^*, b^*)$ với weight nhỏ $\gamma$.
3. **Fully-corrective weight update (tùy chọn nhưng “đúng theory”).**
    Giữ $(v_i,A_i,b_i)$ cố định, solve lại weights $w$ (và/hoặc opacity) bằng một subproblem (least squares / coordinate descent).
    Đây giống ADCG step 4(a). [Nick Boyd](https://nboyd.github.io/adcg.pdf)
4. **Sliding step:** local descent trên $(A_i,b_i)$ để “move support” (ADCG step 4(c); SFW step 8). [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
5. **Prune** atoms có $w_i$ nhỏ hoặc contribution thấp (giống ADCG prune).

**Stopping criterion (proxy):** nếu best LMO score $\|g_\mu(A^*,b^*)\|$ < threshold thì dừng densify.

### B6) Complexity (để reviewer không chê “đẹp nhưng không chạy”)

Giả sử mỗi densify event bạn đánh giá $M$ candidates, mỗi candidate dùng $S$ sample points (hoặc patch pixels) để xấp xỉ $g_\mu(A,b)$.

* Chi phí compute residual/one backward: ~ chi phí một training step.
* Chi phí LMO approx: $O(M\cdot S)$ query $\delta \mathcal{L}$ (với voxel grid, mỗi query là O(1) nội suy; với render-based scoring, là O(#pixels trong footprint)).

Key “engineering claim” bạn cần trong paper:

* $M$ có thể nhỏ (vài nghìn) vì lấy từ top-error pixels (Bulò et al. cũng dùng pixel-error driven). [arXiv](https://arxiv.org/abs/2404.06109)
* $S$ nhỏ vì Gaussian footprint nhỏ.
* Densify không chạy mỗi iter, chỉ mỗi $T$.

---

## C. “Theory” (paper-ready): Proxy convex problem + liên hệ trực tiếp SRM Theorem 1/Prop 1

Đây là phần reviewer sẽ soi: bạn có _thực sự_ dựa vào theorem hay chỉ trích dẫn cho vui.

### C1) Proxy convex formulation: BLASSO/measure regression trên splat atoms

Để có theory chuẩn, bạn phải tách “pipeline 3DGS thực” (nonconvex) khỏi “proxy” (convex) mà bạn dùng để justify thuật toán.

**Proxy assumption:** ta xét một forward operator $\Phi$ **tuyến tính** theo field $f$ (ví dụ linearized rendering / tomography / Laplace-like operator), và dùng squared loss:

$$F(f)=\frac12\|\Phi f - y\|_2^2.$$

SRM đã đưa dạng inverse problem tổng quát và cách tính $\delta F$ bằng linearization + adjoint:

$$\delta F[f] = (D_f\Phi)^*(\Phi f - y).$$

Trong trường hợp $\Phi$ tuyến tính, $D_f\Phi=\Phi$.

Ta tối ưu trên measure $m$ (không nhất thiết xác suất) với total variation (TV) penalty hoặc constraint:

* **Constrained:** $|m|(\Theta)\le \tau$
* **Regularized (BLASSO):**

$$\min_{m}\ \frac12\|\Phi(f_m)-y\|_2^2 + \lambda |m|(\Theta).$$

Đây đúng là setting của **BLASSO** và **Sliding Frank–Wolfe**. [EPFL Biomedical Imaging Group](https://bigwww.epfl.ch/publications/denoyelle1901.pdf)

### C2) LMO c

Splat Regression Models

úc với SRM Theorem 1

Với $F(f)=\frac12\|\Phi f-y\|^2$, ta có:

$$\delta F[f]=\Phi^*(\Phi f-y).$$

Và LMO chọn atom tối đa hóa tương quan với gradient (giống ADCG “arg max |<ψ(θ),g>|”). [Nick Boyd](https://nboyd.github.io/adcg.pdf)

Ở đây, “feature map” $\psi(\theta)$ chính là $x\mapsto v\rho_{A,b}(x)$ đi qua $\Phi$. Score là:

$$\langle v, \int \delta F(x)\rho_{A,b}(x)\pi(dx)\rangle,$$

đúng như SRM Theorem 1 cho $\nabla\nabla_v$.

=> Liên hệ theorem là **direct**, không phải “mượn danh”.

### C3) Guarantee nào là “thật” (đã có theorem), và cái nào là “hy vọng”?

**(i) Convergence của Frank–Wolfe cho bài toán convex smooth:**
FW có rate $O(1/k)$ và có duality gap certificate (Jaggi). [Infoscience+1](https://infoscience.epfl.ch/server/api/core/bitstreams/eab3824d-34c3-4014-be75-58e3aad8dc35/content)

**(ii) ADCG guarantee:**
ADCG là CGM + local descent; paper nhấn mạnh: thuật toán có thể hiểu là alternating giữa bước descent trên bài

Splat Regression Models

ều và bước descent finite-dimensional nonconvex; bound CGM vẫn dùng làm stopping condition. [Nick Boyd](https://nboyd.github.io/adcg.pdf)

**(iii) Sliding Frank–Wolfe (SFW) finite termination cho BLASSO:**
SFW chứng minh dưới giả thiết “dual certificate nondegenerate” thì thuật toán recover nghiệm sau hữu hạn bước. [EPFL Biomedical Imaging Group](https://bigwww.epfl.ch/publications/denoyelle1901.pdf)

**Điểm quan trọng để không bị reviewer bắt bẻ:**
Các guarantee (i)-(iii) **chỉ áp dụng cho proxy convex** (linear measurement + convex loss + TV constraint/penalty).

Trong 3DGS thực:

* renderer $\mathcal{R}$ không tuyến tính
* occlusion/alpha gây nonconvexity
    => bạn không thể claim convergence-to-global-optimum cho NVS thực (nếu claim, reviewer sẽ “đập” ngay).

Điều bạn có thể claim một cách trung thực:

* “Our densification step is the LMO of the convex proxy; empirically it provides a principled alternative to heuristic split/clone.”
* “We retain a descent property by line-search / accepting only atoms that reduce the true loss.”

### C4) Vai trò của Proposition 1 (Bures–Wasserstein) trong “Sliding” step

Nếu bạn muốn _theory-based sliding_ (khác với “chạy Adam như cũ”), bạn có thể đóng gói như sau:

* Covariance $\Sigma=AA^T$ nằm trên manifold SPD với metric induced bởi $W_2$ giữa Gaussian measures (Bures–Wasserstein). SRM Proposition 1 nêu đúng công thức và tính geodesically convex của space splats.
* Có literature về optimization trên BW manifold (ví dụ Chewi et al. phân tích GD/SGD cho BW barycenters dùng PL inequality). [arXiv+1](https://arxiv.org/abs/2001.01700)
* Bạn có thể định nghĩa “slide” như một bước gradient descent trên manifold (hoặc một retraction-based update) thay vì update Euclidean trong tham số hóa $A$.

Đây là chỗ làm contribution “theory+CV” nghe rất thuyết phục (nếu bạn làm thật).

---

## D. Reviewer analysis: novelty & khả thi của ý tưởng SFW-densify

Mình sẽ review theo tiêu chí kiểu CVPR/ICCV/SIGGRAPH (A*): **novelty, technical soundness, significance, clarity, empirical**.

### D1) Novelty

Splat Regression Models

S densification

Hiện tại densification của 3DGS chủ yếu là heuristic và “tuning-heavy”, và nhiều paper gần đây đang cố “principle hóa” tiêu chí densify, ví dụ pixel-error driven densification và kiểm soát số primitives. [arXiv](https://arxiv.org/abs/2404.06109)
Một số hướng khác thì “atomize” hoặc thay đổi logic split/clone nhằm tăng chi tiết và ổn định. [arXiv+1](https://arxiv.org/abs/2405.12369)

Trong search mình làm, mình không thấy paper nào “đóng gói densification = Frank–Wolfe LMO trên measure của Gaussians” một cách trực tiếp (các kết quả Frank–Wolfe xuất hiện chủ yếu ở bài toán tối ưu khác, không phải densification của 3DGS).

=> **Novelty: khá tốt**, đặc biệt vì bạn mang vào một khung measure-optimization có truyền thống mạnh (ADCG/SFW). [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)

### D2) Technical soundness: có đang dựa trên theory SRM thật không?

**Điểm cộng rất lớn:**

* Densification-step bạn đề xuất có thể derive “sạch” từ SRM Definition 2 (first variation) và liên hệ trực tiếp với SRM Theorem 1 qua $g_\mu(A,b)=\int \delta F\rho_{A,b}$.
* Pattern “add atom + slide support” đúng y chang ADCG và SFW. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)

**Điểm reviewer sẽ bắt:**

* Bạn phải **ràng buộc $v$** (và range của $A$, $b$) để miền là compact; nếu không FW formalization bị hỏng.
* Bạn phải nói rõ: guarantee chỉ cho proxy convex; còn NVS thực chỉ là heuristic được “motivated by theory”.
* Dấu $\nabla_A$ trong SRM (statement vs proof) nếu bạn dùng, bạn phải tự derive và fix (mình đã chỉ ở A6).

=> **Soundness: đạt**, nếu bạn viết paper trung thực (không overclaim) và formalize domain đúng.

### D3) Feasibility: có implement được không hay sẽ “stuck”?

Điểm nghẽn 100% nằm ở **LMO approximat

Splat Regression Models

o

Splat Regression Models

athcal{L}(x)) 3D field), bạn cần một kỹ thuật backprojection/adjoint đủ tốt và không quá tốn. Đây là phần tốn công nhưng “đúng” và có thể biến thành contribution.

* Nếu bạn chọn Route 2 (candidate + render-based score), bạn implement nhanh hơn, nhưng phải chứng minh “approx-LMO” đủ tốt (thí dụ ablation: candidate size vs quality; score monotonicity).

Theo mình, để tránh stuck:

* Bắt đầu Route 2 để ra kết quả nhanh
* Sau đó nâng cấp Route 1 như một “theory-consistent variant”

    Splat Regression Models

    n)

Về runtime: densify không chạy mỗi step; candidate scoring có thể giới hạn bởi top-error pixels (đây cũng là spirit của Bulò et al.). [arXiv](https://arxiv.org/abs/2404.06109)

=> **Feasibility: khá**, với điều kiện bạn không cố làm “LMO continuous exact” ngay từ ngày đầu.

### D4) Significance: reviewer có thấy “đáng publish” không?

Bạn sẽ thuyết phục reviewer nếu chứng minh được ít nhất một trong các “đinh” sau:

1. **Chất lượng tốt hơn với ít Gaussians hơn** (compression / sparsity) nhờ “greedy sparse selection” bản chất của FW. (Jaggi nhấn mạnh FW iterates là sparse combination của extreme points.) [Infoscience](https://infoscience.epfl.ch/server/api/core/bitstreams/eab3824d-34c3-4014-be75-58e3aad8dc35/content)
2. **Training ổn định hơn** (ít phụ thuộc hyperparameter densify, ít artifact do clone/split).
3. Có **stopping criterion kiểu dual gap** (proxy) thay vì heuristic “densify đến step 15k”. (ADCG dùng conditional gradient bound làm termination.) [Nick Boyd](https://nboyd.github.io/adcg.pdf)

Nếu paper chỉ “đổi tiêu chí densify” mà gain nhỏ, reviewer sẽ xếp vào nhóm incremental như nhiều paper densification khác.

---

## E. Kết luận thực dụng: có nên theo hướng này để nhắm A* không?

Mình đánh giá: **nên theo**, và đây là phiên bản “đủ chắc” để không bị stuck:

**Hướng mạnh nhất (đề xuất theo đuổi):**

> _SFW-densify for 3DGS_: xem densification như Frank–Wolfe LMO trên measure của Gaussians, kết hợp “sliding” (ADCG/SFW style) + (tùy chọn) manifold-aware covariance update dựa trên Bures–Wasserstein (SRM Proposition 1).

**Vì sao hướng này “đúng” và không bịa:**

* Derive trực tiếp từ SRM Definition 2 và gắn thẳng vào SRM Theorem 1.
* Có precedent mạnh từ ADCG và Sliding Frank–Wolfe trên measure. [Nick Boyd+1](https://nboyd.github.io/adcg.pdf)
* Có “theory story” đủ thuyết phục: proxy convex + inexact oracle + local descent, và bạn nói rõ giới hạn khi qua nonconvex renderer.

**Nếu reviewer reject, khả năng cao vì:**

* bạn overclaim “guarantee cho NVS thực”, hoặc
* bạn không có LMO approximate đủ mạnh/nhanh để chạy benchmark lớn, hoặc
* cải thiện thực nghiệm không đủ lớn so với các densification revision gần đây. [arXiv+1](https://arxiv.org/abs/2404.06109)

---

Nếu bạn muốn bước tiếp theo thật “implementation-consistent”, mình sẽ viết luôn một _design doc_ kiểu: chọn Route 2 trước, định nghĩa score cụ thể trên patch pixels (đúng first-order gain của squared error), cách lấy candidate từ rendered depth + multi-view consistency check, và cách integrate vào code 3DGS mà không tăng wall-clock quá nhiều. Điều này sẽ biế

Splat Regression Models

Splat Regression Models

ry nghe hay nhưng code không chạy”.

* * *

**You:**

Phân tích xem liệu phương pháp có tốt hơn những phương pháp khác ở điểm nào, intution và sell point phương pháp của mình là gì và đang cố gắng giải quyết và đóng góp gì cho cộng đồng. Hãy nghĩ những câu hỏi tại sao và cố gắng trả lời nó, verify lại theory và cả implementation để không bị hallucinate và cẩn thận vấn đề không thể implement được thực tế, vì bạn cần phân tích kĩ càng trước khi làm, cover các case có thể để chắc chắn hơn chứ không phải asusmption. Kiểm tra xem lí thuyết của bạn có “implementation-consistent”, với 3dgs hay không, cách integrate vào code 3DGS mà không tăng wall-clock quá nhiều. Cẩn thận theory trông hay nhưng code lại không chạy. Tốt nhất nên phân tích lại và trình bày dive into the detail pipeline

* * *

**ChatGPT:**

Dưới góc nhìn “paper reviewer khó tính + implementer thực dụng”, mình sẽ phân tích kỹ **Conditional Gradient / Sliding Frank–Wolfe (SFW) Densification** xem:

1. nó **thắng ở đâu** so với densification hiện tại của 3DGS và các biến thể gần đây,
2. **sell point** của mình là gì (vì sao cộng đồng nên quan tâm),
3. **lý thuyết nào chống lưng** (và chỗ nào chỉ là giả định),
4. có **implementation-consistent** với pipeline 3DGS không (không tăng wall-clock vô lý),
5. và bạn phải **đóng gói idea → method → code** sao cho reviewer A* khó bắt bẻ.

Mình sẽ bám sát **3 trụ**: (i) densification gốc của 3DGS, (ii) gap mà các paper densification gần đây đang chữa, (iii) formal hóa theo SRM + Conditional Gradient/ADCG/SFW.

---

## 1) Densification gốc của 3DGS đang làm gì, và nó “gãy” ở đâu?

### Cơ chế gốc (ADC – Adaptive Density Control)

3DGS interleave tối ưu tham số với một thủ tục tăng/giảm số Gaussian. Tóm tắt đúng ý chính của paper:

* Họ **tích lũy gradient của vị trí 2D (view-space mean)** cho từng Gaussian qua nhiều views, lấy **average magnitude** làm tín hiệu “thiếu chi tiết → nên densify”. [ar5iv](https://ar5iv.org/pdf/2308.04079)
* Khi densify:
    * Nếu Gaussian **nhỏ** (scale dưới ngưỡng theo scene extent) → **clone** (copy, dịch theo hướng gradient). [ar5iv](https://ar5iv.org/pdf/2308.04079)
    * Nếu Gaussian **lớn** → **split** thành 2 cái, giảm scale và sample vị trí từ Gaussian gốc. [ar5iv](https://ar5iv.org/pdf/2308.04079)
* Có **opacity reset** định kỳ để tránh “kẹt” tối ưu, rồi **prune** những Gaussian opacity thấp và/hoặc footprint quá lớn. [ar5iv](https://ar5iv.org/pdf/2308.04079)

### Những vấn đề “hệ thống” của ADC mà cộng đồng đang chữa

Các paper gần đây chỉ ra khá rõ (và đây là chỗ gap của hướng densification):

* **Threshold dựa trên magnitude gradient rất nhạy**: đổi loss, đổi lr, đổi render setting là phải tune lại; tiêu chí không “intuitive”. [arXiv](https://arxiv.org/html/2404.06109v1)
* Có tình huống **underfitting nhưng gradient vẫn dưới ngưỡng** → ADC không grow, gây blur ở vùng texture/high-frequency. [arXiv+1](https://arxiv.org/html/2404.06109v1)
* **Gradient collision**: khi Gaussian phủ nhiều pixel, gradient từ các pixel có thể triệt tiêu nhau (hướng khác nhau) → tổng gradient nhỏ → không split được dù cần. AbsGS phân tích đúng điểm này. [ty424.github.io](https://ty424.github.io/AbsGS.github.io/)
* **Thiếu điểm khởi tạo / coverage**: Pixel-GS nói rõ: chỉ dùng average gradient khiến vùng thiếu init points bị blur/needle-like artifacts. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
* Thêm nữa: floaters near camera, bias opacity handling khi cloning, kiểm soát số primitive… đều là “triệu chứng” của việc densification là heuristic. [arXiv+1](https://arxiv.org/html/2404.06109v1)

=> Nói kiểu “reviewer”: ADC không sai, nhưng nó **là heuristic pipeline**. Cộng đồng đang cố “patch” tiêu chí densify (pixel-aware, abs-gradient, edge-aware, perceptual-aware…) thay vì đổi “bản chất tối ưu hoá”.

---

## 2) Vậy “Conditional Gradient / Sliding Frank–Wolfe densification” tốt hơn ở điểm nào?

### Sell point (điểm bán hàng) nếu bạn viết paper A*

**Câu chuyện bạn muốn bán cho reviewer**:

> Densification của 3DGS thực chất là một bài toán **chọn thêm atom/basis function** (mỗi Gaussian là một atom) để giảm loss. ADC đang làm điều đó bằng rule-of-thumb.
> Mình thay rule-of-thumb bằng một thủ tục **greedy nhưng có nền tối ưu hoá đo lường (measure optimization)**: Conditional Gradient / Frank–Wolfe, và thêm “sliding” (local descent) để khớp với tính không lồi và geometry của Gaussian parameters.

Điểm mạnh cụ thể:

1. **Principled “where/when to densify”**
    Thay vì “gradient magnitude > τpos”, bạn dùng tiêu chí “atom nào đem lại giảm loss lớn nhất theo linearization” — đây đúng tinh thần Frank–Wolfe / ADCG. [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v28/jaggi13.pdf)
2. **Có certificate/stopping proxy**
    Frank–Wolfe/CGM có khái niệm dual gap / lower bound trong convex setting; ADCG dùng điều này làm stopping condition + bounded memory. [arXiv](https://arxiv.org/pdf/1507.01562)
    Trong 3DGS, dù full objective không convex, bạn vẫn có thể dùng **proxy convex** để có “logic dừng” hợp lý (reviewer rất thích: “không phải heuristic vô cớ”).
3. **Giải thích thống nhất nhiều patch hiện có**
    Pixel-error driven densification (Bulò et al.) đã đi theo hướng “pixel residual → densify” [arXiv](https://arxiv.org/html/2404.06109v1), nhưng vẫn thiếu một khung “atom selection = optimization step”. FW cung cấp khung đó.
    AbsGS sửa gradient collision bằng abs-sum [ty424.github.io](https://ty424.github.io/AbsGS.github.io/); trong FW score bạn có thể thay norm/aggregator (L1-like) một cách có lý giải: bạn đang đổi geometry/penalty trong proxy.
4. **Khả năng giảm số Gaussian hoặc tăng quality ở cùng budget**
    Nếu oracle chọn atom “đúng chỗ”, bạn có thể đạt quality với ít Gaussians hơn (đây là điểm reviewer computer vision/graphics cực quan tâm: quality–efficiency trade-off). Các paper densification mới cũng hay claim giảm số primitive, nhưng họ vẫn dựa threshold. [arXiv+1](https://arxiv.org/html/2506.12400v1)
5. **Implementation-consistent nếu bạn thiết kế LMO đúng kiểu 3DGS**
    Bạn không cần giải global optimization; bạn cần LMO “xấp xỉ đủ tốt” + chỉ chạy theo interval giống ADC.

---

## 3) Lý thuyết SRM chống lưng bạn như thế nào (và cái gì KHÔNG được bịa)?

Bạn yêu cầu “đừng hallucinate”: OK, mình chốt rõ những mệnh đề nào “có theorem”, mệnh đề nào “proxy/assumption”.

### 3.1 SRM định nghĩa splat model & gradient (đây là nền thật)

SRM đưa mô hình splat regression:

$$f_\mu(x)=\int v\,\rho_{A,b}(x)\,\mu(dv,dA,db)$$

(đây là dạng “Gaussian mixture layer” / “splat measure”).

**Theorem 1 của SRM** cho gradient Wasserstein–Fisher–Rao của $\mu\mapsto F(f_\mu)$, trong đó quan trọng nhất với densification là phần Wasserstein gradient theo $v$:

$$\nabla_v F(f_\mu)(v,A,b)=\int \delta F(x)\,\rho_{A,b}(x)\,\pi(dx).$$

và các gradient theo $A,b$ cũng có công thức tích phân theo $\delta F$ và $\nabla \log \rho_{A,b}$.

**Diễn giải cực quan trọng**:

* (\delta F(

    Splat Regression Models

    tion** (error signal) — SRM nói rõ nó chính là vector lỗi mà backprop trả về trong ERM case, và trong NVS thì phụ thuộc rendering approximation và có thể lấy bằng autodiff/backprop.
* $\nabla_v F$ là **residual được làm trơn** bởi kernel $\rho_{A,b}$. Nó nói “nếu tại location/scale này còn lỗi, thì weight/color gradient sẽ lớn”.

➡️ Đây là mấu chốt để thiết kế **LMO score** cho FW densification mà không bịa.

### 3.2 Cầu nối sang Frank–Wolfe / ADCG / SFW (đây cũng là nền thật)

* Frank–Wolfe/Conditional Gradient là phương pháp tối ưu hoá lồi trên convex hull / atomic sets, mỗi vòng thêm **một atom**. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v28/jaggi13.pdf)
* ADCG (Boyd, Schiebinger, Recht) formalize “sparse inverse problem” trên **không gian measure**: mỗi iteration chọn source mới bằng oracle (argmin linearization), rồi tối ưu weights, rồi local descent (nonconvex) — vừa có guarantee convex, vừa có performance thực tế. [arXiv](https://arxiv.org/pdf/1507.01562)
* Sliding Frank–Wolfe (Denoyelle et al.) cũng interleave “add spike” + “slide support bằng local optimization” và có kết quả lý thuyết mạnh cho BLASSO (trong setting thích hợp). [arXiv+1](https://arxiv.org/abs/1811.06416)

➡️ Vậy việc bạn đề xuất “add Gaussian atom + sliding refine” không hề là tưởng tượng; nó có tiền lệ lý thuyết chắc chắn trong **measure sparse optimization**.

### 3.3 Chỗ nào là proxy/assumption (bạn phải nói thẳng trong paper)

* Full 3DGS objective dùng **α-blending + occlusion** → mapping từ $\mu$ đến image **không tuyến tính** và không convex. SRM cũng nói gradient phụ thuộc approximation và computed via autodiff.
* Vì vậy, “global optimality / dual gap guarantee” chỉ đúng nếu bạn chọn **proxy convex formulation** (ví dụ linearized/additive rendering hoặc fix transmittance), hoặc nếu bạn chỉ claim “descent heuristic with theory-guided scoring”, không claim global optimum cho full pipeline.

Nếu bạn viết paper mà “pretend convex” cho full α-blend thì reviewer sẽ đập.

---

## 4) Làm sao để lý thuyết “implementation-consistent” với pipeline 3DGS?

Đây là phần quan trọng nhất để tránh “theory hay nhưng code không chạy”.

### 4.1 Nhìn pip

Splat Regression Models

esidual + atom”

Trong 3DGS training, mỗi iteration bạn đã có:

* render $\hat I$ từ current Gaussians
* loss $\mathcal L(\hat I, I)$
* backprop ra **residual per pixel** (gradient wrt rendered color)
* backprop tiếp ra gradients wrt Gaussian params (xyz, scale, rot, opacity, SH).

ADC hiện tại chỉ lấy một “summary statistic”: average magnitude của view-space position gradients. [ar5iv](https://ar5iv.org/pdf/2308.04079)

**FW densification** sẽ cố dùng residual đó một cách “đúng bài”: chọn atom mới sao cho **inner product với negative gradient lớn nhất** (steepest descent trong atomic dictionary).

### 4.2 Proxy convex objective để có “Theory” sạch

Bạn cần một objective convex “proxy” mà vẫn liên hệ chặt với 3DGS.

Một proxy điển hình và rất “implementation-consistent” là:

* Fix transmittance/visibility theo current scene, coi contribution của một Gaussian mới là **additive nhỏ** (first-order approximation). Khi opacity của atom mới nhỏ, ảnh hưởng đến ordering/transmittance của các atom khác là bậc hai → bỏ qua được trong bước chọn atom. Đây là assumption hợp lý khi bạn **khởi tạo opacity nhỏ** (3DGS cũng hay reset opacity nhỏ). [ar5iv](https://ar5iv.org/pdf/2308.04079)
* Khi đó measurement model gần dạng tuyến tính: $$\hat I \approx I_\text{old} + \Phi(\theta)\,w$$ với $\theta$ là tham số geometry của Gaussian mới (position/covariance), còn $w$ là vector appearance (RGB hoặc SH coeff).
* Dùng squared loss trên mini-batch pixels/views: convex theo $w$ (và theo measure weights nếu bạn constrain atomic norm).

Trong SRM, ERM case δF chính là error signal của loss; trong NVS, gradient tính qua autodiff.

### 4.3 LMO (Linear Minimization Oracle) “đúng kiểu 3DGS”

Trong conditional gradient cho bài toán atomic, LMO dạng:

$$\theta^\star \in \arg\max_{\theta\in\Theta}\; \left\|\langle \psi(\theta), g\rangle\right\|$$

Trong ADCG, bước “compute next source” là argmax correlation với gradient. [arXiv](https://arxiv.org/pdf/1507.01562)

**Trong 3DGS**, một lựa chọn oracle rất thực dụng là:

* Lấy mini-batch views $n=1..B$.
* Với mỗi view $n$, bạn có residual image $r_n(u)=\partial \mathcal L/\partial \hat I_n(u)$ (3-vector RGB).
* Với candidate Gaussian geometry $\theta=(b,\Sigma,\alpha)$, ta tính footprint 2D Gaussian $G_{n,\theta}(u)$ (kernel weight) và một visibility/transmittance factor $T_{n,\theta}(u)$ (có thể xấp xỉ bằng depth buffer / accumulated alpha).
* Score (vector) cho appearance gradient: $$g(\theta)=\sum_{n=1}^B \sum_{u\in\Omega} T_{n,\theta}(u)\,G_{n,\theta}(u)\,r_n(u).$$
* Rồi chọn $\theta^\star = \arg\max \|g(\theta)\| $.

**Đây không bịa**: nó đúng tinh thần SRM Theorem 1 phần $\nabla_v F = \int \delta F \rho_{A,b}$ — chỉ khác là “x” bây giờ là pixel-space sample + operator $\Phi$ (rendering) đưa vào qua $T\cdot G$.

### 4.4 Candidate set để oracle không đốt GPU

Đây là nơi “implementation-consistent” sống còn: bạn **không thể** tối ưu $\theta$ liên tục trên toàn $\mathbb R^3\times$covariance một cách brute-force.

Bạn cần LMO **xấp xỉ** kiểu ADCG: tối ưu correlation bằng cách search trên candidate set hữu hạn hoặc local refinement. [arXiv](https://arxiv.org/pdf/1507.01562)

Hai cách (mình khuyên dùng cách 1 cho phiên bản đầu để chắc chạy):

**(A) Restricted-LMO (drop-in replacement cho ADC)**

* Generate candidates bằng chính các thao tác **clone/split** (và các biến thể như long-axis split) từ một subset Gaussians.
* Nhưng thay vì threshold τpos để chọn split/clone, bạn **score từng candidate bằng $\|g(\theta)\|$** rồi pick top-K.
    => Ưu điểm: không phải solve geometry từ pixel-to-3D; integrate cực dễ vào code densify_and_prune.
    => Nhược: vẫn “local” quanh Gaussians hiện có, ít tạo mới ở vùng hoàn toàn thiếu init.

**(B) Residual-driven LMO (mạnh hơn nhưng khó hơn)**

* Lấy top-M pixels có |residual| lớn (hoặc SSIM error map như Bulò et al. đề xuất). [arXiv](https://arxiv.org/html/2404.06109v1)
* Backproject pixel ra 3D bằng depth estimate từ current render (expected depth) hoặc SfM depth (nếu có).
* Tạo candidate Gaussian ở vị trí đó, covariance từ Jacobian projection (hoặc heuristic scale theo depth).
    => Ưu điểm: có thể “đẻ” Gaussians ở vùng thiếu coverage.
    => Rủi ro: depth không chuẩn → floaters, cần visibility gating rất kỹ (Pixel-GS và các paper khác cũng phải xử lý floaters). [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

---

## 5) “Why” questions mà reviewer sẽ hỏi, và câu trả lời không bị hớ

### Q1: “Tại sao FW densification tốt hơn Pixel-GS / Revising Densification / AbsGS…?”

* Pixel-GS và AbsGS **sửa metric** để _quyết định densify_ tốt hơn nhưng vẫn trong khung “tính một số rồi so threshold”. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
* Revising Densification đi xa hơn: pixel-error driven, và nói rõ threshold gradient magnitude nhạy. [arXiv+1](https://arxiv.org/html/2404.06109v1)
* FW/SFW/ADCG đưa ra **một cách nhìn khác**: densification = **greedy optimization trên atomic dictionary**. Cái bạn “đóng góp” không phải thêm một metric nữa, mà là:
    * một **algorithmic principle** để chọn atom (LMO),
    * một **điều kiện dừng/certificate (trong proxy convex)**,
    * và một cách unify nhiều heuristic (pixel-error, abs-grad, edge-aware) thành weight trong score.
        Điều này align rất tốt với văn học ADCG/SFW: add atom + local descent. [arXiv+2arXiv+2](https://arxiv.org/pdf/1507.01562)

=> Reviewer sẽ chấp nhận nếu bạn chứng minh: “các heuristics hiện có tương ứng với các biến thể của score / candidate generation”, và FW là wrapper thống nhất.

### Q2: “Tại sao phải theory-based? Có thật SRM giúp gì không?”

SRM cung cấp đúng 2 thứ quan trọng cho bạn:

1. **Ngôn ngữ measure / splat model**: 3DGS được diễn giải như một instance của splat regression modeling, và rendering được đưa vào như operator $A$ (RTE/α-blending) — gradient tính qua autodiff.
2. **Công thức gradient theo measure parameters (Theorem 1)**: đặc biệt phần $\nabla_v F = \int \delta F \rho$ cho ta một score “tự nhiên” để đo “ở đâu còn residual”.

Nếu không có SRM, bạn vẫn làm được FW heuristic; nhưng SRM giúp bạn **justify** rằng score bạn dùng không phải bịa mà là “smoothed first variation”.

### Q3: “Nhưng α-blending không lồi, vậy certificate của bạn có nghĩa không?”

Đúng: full objective không lồi. Bạn phải trung thực và làm như ADCG/SFW thường làm:

* **Theory section**: đặt một proxy convex (linearized measurement / fixed transmittance) để chứng minh property kiểu FW: monotonic decrease, sublinear rate, dual gap… dựa trên Jaggi/ADCG/SFW. [Proceedings of Machine Learning Research+2arXiv+2](https://proceedings.mlr.press/v28/jaggi13.pdf)
* **Method/Implementation section**: nói rõ trong pipeline thật, bạn dùng oracle score từ proxy nhưng **accept/reject theo loss thật** (mini-batch line search). ADCG cũng nhấn mạnh bước final update có thể chọn điểm tốt hơn mà vẫn giữ guarantee trong convex case. [arXiv](https://arxiv.org/pdf/1507.01562)

Nếu bạn làm vậy, reviewer thường sẽ ok: “proxy theory + real-world algorithm”.

---

## 6) Phần “Method” paper-ready (nhưng chú trọng sell + implementation)

Mình sẽ viết một bản “đủ paper-ready” (không quá dài) đúng những gì reviewer CV/graphics hay soi: objective, oracle, algorithm, complexity, và theory link tới SRM Theorem 1 + ADCG/SFW.

### 6.1 Splat-measure parameterization

Ta mô hình hóa trường $f_\mu$ dưới dạng splat measure (SRM):

$$f_\mu(x)=\int v \,\rho_{A,b}(x)\,\mu(dv,dA,db),$$

trong đó $\rho_{A,b}$ là density của “mother splat” được affine-transform (ví dụ Gaussian với covariance $AA^\top$).

Trong NVS, SRM mô tả cách 3DGS parameterize extinction $\sigma(x)=g_\nu(x)$ và emission $s(x,\cdot)$ bằng spherical harmonics với $f_\mu$ là splat model; rendering xấp xỉ bằng α-blending và gradient lấy bằng autodiff.

### 6.2 Objective

Ta tối ưu:

$$\min_{\mu\in\mathcal M} \; \mathcal L(\mu)
:=\frac{1}{B}\sum_{n=1}^B \ell(\mathrm{Render}_{c_n}(\mu), I_{c_n})
\;+\;\lambda \,\Omega(\mu),$$

với $\Omega$ có thể là ràng buộc “budget” (ví dụ total mass / atomic norm / hoặc đơn giản là constraint số Gaussian tối đa).

**Proxy convex** (phục vụ Theory & oracle):
Tại iterate $\mu_t$, linearize rendering wrt **một atom mới** $\theta$, coi contribution là additive nhỏ:

$$\mathrm{Render}(\mu_t + w\delta_\theta)\approx \mathrm{Render}(\mu_t) + w\,\Phi_t(\theta),$$

với $\Phi_t(\theta)$ là “image footprint featur

Splat Regression Models

visibility hiện tại (fixed transmittance). Khi dùng squared loss, objective là convex theo $w$ và theo measure weights với atomic constraint.

### 6.3 LMO (Linear Minimization Oracle) xấp xỉ

Gọi r_n(u)=\partial \ell/\partial:contentReference[oaicite:41]{index=41}dual gradient tại pixel \(u ở view $n$.
Với candidate geometry $\theta$, define:

$$g_t(\theta)=\sum_{n=1}^B\sum_{u\in\Omega}
T_{t,n,\theta}(u)\,G_{t,n,\theta}(u)\,r_n(u),$$

trong đó $G$ là 2D Gaussian footprint (từ project covariance như 3DGS), và $T$ là transmittance/visibility weight (xấp xỉ từ accumulated alpha buffer).

Ta chọn:

$$\theta_t \in \arg\max_{\theta\in\Theta_{\text{cand}}}\|g_t(\theta)\|_2,$$

và khởi tạo appearance (RGB/SH coeff) theo hướng giảm loss, ví dụ $w_t\propto -g_t(\theta_t)$ (line search hoặc normalize).

**Liên hệ trực tiếp SRM Theorem 1**: trong SRM, gradient wrt weight/feature $v$ là $\nabla_v F = \int \delta F(x)\rho_{A,b}(x)\pi(dx)$.
Ở đây $g_t(\theta)$ chính là “smoothed first variation” sau khi đưa rendering operator vào (proxy linearization), nên là một LMO score nhất quán với measure-gradient.

### 6.4 Algorithm: Sliding Frank–Wolfe Densification for 3DGS

**Input**: current Gaussians $\mu_t$, candidate set $\Theta_\text{cand}$, budget K atoms per densify step.

**For** each densification interval:

1. Render mini-batch views, compute loss, backprop lấy residual $r_n(u)$.
2. **(LMO)** Score all candidates $\theta\in\Theta_\text{cand}$ bằng $\|g_t(\theta)\|$, pick top-K.
3. **(FW step)** Add K new atoms với opacity nhỏ, init color từ GT pixel hoặc từ $-g_t$.
    Có thể làm line search trên mini-batch để chọn step-size (hoặc dùng $\gamma=2/(k+2)$ như FW cổ điển). [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v28/jaggi13.pdf)
4. **(Sliding / local descent)** Chạy vài bước Adam/SGD bình thường để refine toàn bộ params; hoặc chạy local descent trên support mới (ADCG style). [arXiv+1](https://arxiv.org/pdf/1507.01562)
5. Prune theo opacity/footprint như 3DGS (giữ nguyên để không phá pipeline). [ar5iv](https://ar5iv.org/pdf/2308.04079)

### 6.5 Complexity & wall-clock

* ADC gốc: densify dựa trên per-Gaussian gradient statistic → rất rẻ. [ar5iv](https://ar5iv.org/pdf/2308.04079)
* FW-densify thêm chi phí scoring:
    * Nếu dùng **Restricted-LMO**: số candidates ~ O(#Gaussians sampled) và mỗi candidate chỉ cần footprint nhỏ trên vài tiles/pixels → implement được bằng CUDA kernel kiểu rasterize “candidate-only” (không cần full rerender scene).
    * Nếu dùng **Residual-driven**: chi phí thêm ở step backproject + cluster; vẫn chỉ chạy mỗi densify interval (ví dụ 1000 steps).
* Quan trọng: bạn **không được** render lại full scene cho từng candidate. Cách an toàn là:
    * chỉ score trên top-M pixels/patches,
    * chỉ dùng B nhỏ views (1–4),
    * và dùng tile binning giống rasterizer.

---

## 7) Rủi ro thực tế (những cách “code sẽ chết”) và cách né

### Risk A: “LMO cần transmittance T(u), tính thế nào cho đúng?”

* Nếu bạn muốn chính xác như α-blending, T phụ thuộc ordering và opacity của toàn bộ Gaussians.
* Nhưng bạn **chỉ cần proxy**: lấy accumulated alpha/transmittance buffer từ forward pass hiện tại (nhiều implementation có sẵn depth/alpha accum).
* Thêm trick: initialize opacity của atom mới rất nhỏ → assumption “không làm đổi T của các atom khác” tốt hơn.

=> Đây là assumption nhưng **implementation-consistent** và phù hợp với cách 3DGS reset opacity nhỏ để tái tối ưu. [ar5iv](https://ar5iv.org/pdf/2308.04079)

### Risk B: “Oracle chọn atom ở chỗ residual lớn nhưng đó là texture, không phải geometry”

Revising Densification và Perceptual-GS đều nhắc rằng pixel error có thể do texture/perception chứ không do thiếu primitive. [arXiv+1](https://arxiv.org/html/2404.06109v1)

Mitigation:

* Score dùng $r_n(u)$ của photometric loss (RGB) có thể gây bias.
* Bạn nên dùng **structured residual**: edge/gradient magnitude, SSIM error map, hoặc multi-view consistency residual (như Bulò et al. đề xuất propagate pixel error). [arXiv](https://arxiv.org/html/2404.06109v1)
* Trong FW framework, đây chỉ là **weighting** của residual: $r \leftarrow w(u)\,r$. Bạn không phải thay algorithm.

### Risk C: Floaters / near-camera artifacts

Pixel-GS thêm trick scale gradient theo distance để suppress floaters near camera. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
Trong FW score, bạn có thể:

* penalize candidates quá gần camera,
* yêu cầu candidate “được support” bởi ≥2 views (triangulation consistency),
* hoặc require depth agreement với depth buffer.

### Risk D: Overhead scoring quá lớn

Đây là nơi hầu hết “theory-driven densification” chết.

Cách giữ wall-clock:

* Densify interval thưa (giống ADC).
* Candidate count nhỏ:
    * Restricted-LMO: sample subset Gaussians có error cao (theo per-Gaussian accumulated error như Bulò et al. làm). [arXiv](https://arxiv.org/html/2404.06109v1)
    * Residual-driven: top-M pixels/patches, cluster lại.
* Score kernel cực đơn giản: dot(residual, footprint) trên patch/tile.

Nếu bạn làm được overhead < ~10–15% training time, reviewer graphics thường chấp nhận, nhất là nếu #Gaussians giảm hoặc quality tăng.

---

## 8) Chốt lại: phương pháp của bạn đóng góp gì cho cộng đồng 3DGS?

Mình sẽ “đóng hộp” contribution theo kiểu reviewer thích:

1. **A new view:** Densification = **atomic measure optimization** (not heuristic thresholding).
2. **A new algorithmic primitive:** FW/ADCG/SFW-style **oracle-based densification** cho Gaussian splats. [arXiv+2arXiv+2](https://arxiv.org/pdf/1507.01562)
3. **A theory bridge:** dùng SRM Theorem 1 (first variation smoothed by splat) để derive score nhất quán với gradient flow calculus.
4. **A practical plugin:** drop-in thay thế phần “which Gaussian to densify” và/hoặc “where to add new Gaussians”, giữ prune/reset giống 3DGS để ổn định. [ar5iv](https://ar5iv.org/pdf/2308.04079)
5. **Unified explanation** cho nhiều heuristic gần đây: pixel-error, abs-grad, perceptual weighting là các biến thể của residual weighting / candidate generation.

---

## 9) Nếu mình là reviewer A*: mình sẽ đánh giá ý tưởng này thế nào?

### Điểm cộng

* **Novelty có cơ sở**: mình không thấy (qua literature densification phổ biến) ai formalize densification như FW/ADCG measure-optimization cho 3DGS một cách rõ ràng; đa số vẫn patch ADC. (Bạn vẫn phải survey kỹ trong related work của mình, nhưng hiện trạng đúng là vậy.) [arXiv+2ty424.github.io+2](https://arxiv.org/html/2404.06109v1)
* **Theory không phải “math for decoration”**: SRM thật sự cung cấp công thức gradient theo measure params; ADCG/SFW là tiền lệ mạnh cho add-atom+slide.
* **Potential impact**: nếu giảm hyperparameter sensitivity (threshold) và cải thiện quality–#Gaussian trade-off, đây là đóng góp thực tế.

### Điểm mình sẽ bắt bẻ (và bạn phải chuẩn bị)

1. “LMO của bạn có thật sự chọn đúng atom, hay chỉ là một heuristic khác?”
    → Bạn phải cho thấy score = inner product với residual (gradient) theo linearization, và ablation chứng minh nó robust hơn threshold.
2. “Proxy convex theory liên quan gì tới pipeline α-blending thật?”
    → Bạn phải viết rõ: theory cho proxy + accept/reject theo loss thật + empirical evidence.
3. “Overhead bao nhiêu? có chạy real-time training được không?”
    → Bạn phải báo wall-clock, #CUDA kernels thêm, scaling theo #Gaussians.
4. “So sánh với các densification SOTA” (Pixel-GS, AbsGS, Revising Densification, Perceptual-GS…)
    → Nếu chỉ so với vanilla 3DGS thì reject.

---

## Kết luận thực dụng: hướng này có đáng theo đuổi không?

**Có** — nếu bạn commit làm theo “phiên bản chắc chạy” trước:

* bắt đầu bằng **Restricted-LMO** (candidates = split/clone proposals) → thay threshold bằng FW-score;
* sau đó mới mở rộng sang residual-driven spawning.

Đây là cách đi từ idea → implementation mà ít “kẹt” nhất: bạn thay đúng **một module** (decision logic) nhưng giữ pipeline ổn định.

Về novelty: nó đủ “A*-potential” nếu bạn chứng minh được ít nhất 2 trong 3:

* quality tăng rõ rệt ở high-frequency regions (giải quyết failure case kinh điển của ADC),
* hoặc giảm #Gaussians / memory đáng kể ở cùng quality,
* hoặc giảm sensitivity với hyperparameters (threshold-free hoặc threshold ít nhạy) — đúng vấn đề mà Bulò et al. chỉ ra. [arXiv+1](https://arxiv.org/html/2404.06109v1)

Và quan trọng: bạn “bán” được câu chuyện **theory → algorithm → plug-in** dựa trên SRM + ADCG/SFW, không phải “một metric mới”.

---

Nếu bạn muốn bước tiếp theo theo kiểu _không bị stuck_ thì logical nhất là: mình viết tiếp một “Implementation Notes” dạng checklist (mapping đúng vào gaussian-splatting training loop: lấy residual ở đâu, tile binning score kernel ra sao, data structures cần thêm gì, và cách không phá backprop graph). Nhưng ở đây mình dừng ở mức reviewer-grade reasoning + method-ready, để đảm bảo ta đang đi đúng hướng về mặt khoa học và khả thi.

* * *

**You:**

viết tiếp một “Implementation Notes” dạng checklist (mapping đúng vào gaussian-splatting training loop: lấy residual ở đâu, tile binning score kernel ra sao, data structures cần thêm gì, và cách không phá backprop graph) để đảm bảo ta đang đi đúng hướng về mặt khoa học và khả thi.

* * *

**ChatGPT:**

Dưới đây là **“Implementation Notes” dạng checklist** để gắn _Sliding Frank–Wolfe / Conditional-Gradient densification_ vào đúng **training loop của repo graphdeco-inria/gaussian-splatting** (baseline 3DGS). Mình sẽ bám sát các “điểm neo” thật trong code/paper để tránh kiểu theory đẹp nhưng code không chạy.

---

## Implementation Notes (Checklist) — mapping 1:1 vào 3DGS training loop

### 0) Ground truth về pipeline hiện tại (để khỏi cắm nhầm chỗ)

* [ ]  **Nhớ rằng `viewspace_points` trong render_pkg KHÔNG chứa 2D means thật.**
    Trong `gaussian_renderer/__init__.py`, tác giả tạo `screenspace_points = torch.zeros_like(..., requires_grad=True)` _chỉ để lấy gradient wrt 2D means_, rồi truyền nó vào rasterizer như `means2D`. Tensor này về giá trị **vẫn là 0**, nhưng `.grad` sau backward sẽ có. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)
    → Hệ quả: nếu bạn định “tile binning theo means2D từ viewspace_points”, bạn sẽ bin sai (toàn về tile (0,0)).
* [ ]  Rasterizer của 3DGS là tile-based: **tile 16×16**, mỗi Gaussian được “duplicate” cho mọi tile nó overlap; rồi sort theo (tile_id, depth). Paper mô tả rõ cơ chế: key có **higher bits = tile index**, **lower 32 bits = depth**; sau đó `IdentifyTileRanges` để biết range per-tile. [ar5iv](https://ar5iv.org/pdf/2308.04079)
    → Đây là đúng mental model để thiết kế “score kernel” theo tile, và cũng gợi ý đường tối ưu: _tận dụng tileRanges/point_list nếu patch rasterizer_.
* [ ]  `render()` trả về: `rendered_image`, `radii`, `depth_image`, và `visibility_filter = (radii>0).nonzero()` (indices) [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)
    `train.py` dùng `visible = radii > 0` cho SparseAdam path. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)
    → Cẩn thận: `visibility_filter` là indices (M×1), còn `visible` là bool mask (N,).

---

### 1) “Residual” lấy ở đâu (theory-consistent vs proxy)

Trong SRM, Theorem 1 nói gradient theo “amplitude” v của splat phụ thuộc vào tích phân **δF(x) ρA,b(x)**; tức muốn làm LMO/CG đúng nghĩa thì cần **first-variation δF** của objective theo output function.

Với 3DGS, ta có 2 lựa chọn:

#### 1A) Residual **theory-consistent**: dùng gradient của loss theo ảnh render (`∂L/∂I`)

* [ ]  Trong `train.py`, **trước** `loss.backward()` hãy gọi `image.retain_grad()` _chỉ ở những iteration cần densify_ để giảm overhead.
    Lý do: `image` là tensor không-leaf (output của CUDA rasterizer), nếu khô

    Splat Regression Models

    ì `image.grad` sẽ `None`.
* [ ]  Sau `loss.backward()` (nhưng **trước** `optimizer.zero_grad()`), lấy:
    * `residual_img = image.grad.detach()` (shape `[3,H,W]`)
    * dùng nó làm xấp xỉ δF trên domain pixel.
        (Trong SRM notation: δF(x) là first variation; ở đây “x” là pixel coordinate / sample point trong π).

**Ưu:** đúng “δF-based” theo Theorem 1 nên sell tốt với reviewer.
**Nhược:** giữ thêm grad buffer ảnh; nhưng nếu chỉ làm mỗi `densification_interval` thì thường ổn.

#### 1B) Residual **proxy rẻ**: dùng error map `|I - I_gt|` (không cần grad)

* [ ]  `residual_img = (image.detach() - gt.detach()).abs()` hoặc energy norm.
    **Ưu:**

    Splat Regression Models

    view-sampling.
    **Nhược:** không còn đúng δF, bạn phải nói rõ đây là proxy (không claim theory chặt như 1A).

> Recommendation thực dụng: **Prototype nhanh bằng 1B**, rồi “paper version” chuyển sang 1A (hoặc hybrid: 1B tích luỹ theo thời gian + 1A snapshot tại densify step).

---

### 2) Tile residual aggregation (cái bạn feed cho score kernel)

Đặt `T = 16` đúng như paper tile size. [ar5iv](https://ar5iv.org/pdf/2308.04079)

* [ ]  Từ `residual_img` (3×H×W), tạo tile statistics trên GPU:
    * Vector tile residual: `R_tile ∈ ℝ^{3×H_t×W_t}`
        dùng `avg_pool2d(residual_img[None], kernel=T, stride=T)`
    * Hoặc energy scalar: `E_tile ∈ ℝ^{H_t×W_t}`
        `E = ||residual_img||_2` rồi pool.
* [ ]  Flatten cho kernel: `E_tile_flat ∈ ℝ^{N_tiles}` với `tile_id = ty*W_t + tx`.
* [ ]  Optional: EMA qua nhiều iteration để giảm noise (chi phí nhỏ vì tile grid nhỏ).

---

### 3) “Tile binning” cho **score kernel**: lấy Gaussian→tile overlap ở đâu?

Bạn cần biết Gaussian i overlap tile nào để tính score ≈ ∑_{t∈tiles(i)} w_{i,t} · E_tile[t].

**Vấn đề lớn:** repo 3DGS _không trả_ projected means2D thật (vì `means2D` truyền vào là zero placeholder). [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)
Vậy có 2 đường:

#### 3A) Không đụng CUDA: tự project means2D trong PyTorch

* [ ]  Sau backward, trong `torch.no_grad()`:
    * `means3D = gaussians.get_xyz.detach()`
    * Project sang NDC/pixel bằng camera matrices (cùng cái đang đưa vào rasterizer: `world_view_transform`, `full_proj_transform`, image W/H). [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)
    * **Checklist sanity test**: với gaussians visible (`radii>0`), phần lớn projected (x,y) phải nằm trong [0,W)×[0,H). Nếu lệch hàng loạt → bạn đang nhân ma trận sai chiều (row-major/col-major) hoặc y-flip.
        (Đây là “điểm hay chết” nhất nếu chỉ dựa vào cảm giác.)
* [ ]  Lấy `r = radii.detach()`; đây là “screen-space radius” dùng để cull/visible trong pipeline. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)
    (Trong các implementation tile-mapping hiện đại, “radii in pixel unit; radii=0 => invalid” là convention chuẩn. [Gsplat Studio](https://docs.gsplat.studio/main/apis/utils.html))
* [ ]  Từ `(x_i,y_i,r_i)` tính bounding tile range:
    * `tx0 = floor((x-r)/T)`, `tx1 = floor((x+r)/T)`
    * `ty0 = floor((y-r)/T)`, `ty1 = floor((y+r)/T)`
    * clamp vào `[0,W_t-1]×[0,H_t-1]`.

#### 3B) Patch CUDA rasterizer: xuất thẳng “tile intersections / tileRanges”

* [ ]  Paper đã có `DuplicateWithKeys` + `IdentifyTileRanges`. [ar5iv](https://ar5iv.org/pdf/2308.04079)
    Trong code CUDA chắc chắn có các buffer tương tự để rasterize tile-wise. Nếu expose chúng ra PyTorch output:
    * bạn **khỏi phải tự project** + khỏi phải tự compute overlap
    * score kernel có thể chạy “segment reduce” theo tileRanges rất nhanh.
* [ ]  Nếu bạn muốn reference một design đã hệ thống hoá chuyện này: gsplat có API `isect_tiles()` trả ra intersection ids kiểu `image_id | tile_id | depth(32 bits)` và mô tả `radii` là pixel-unit + “0 radii => invalid”. [Gsplat Studio](https://docs.gsplat.studio/main/apis/utils.html)
    → Đây là blueprint cực rõ để bạn implement kernel tương đương trong diff-gaussian-rasterization.

**Tradeoff:** 3B sạch và nhanh hơn, nhưng cần CUDA hacking; 3A nhanh để ra kết quả ngay.

---

### 4) Score kernel (LMO proxy) — implementable, không phá graph

Trong spirit Theorem 1, “điểm” của một splat gắn với δF qua tích phân δF(x)ρ(x).
Ở ảnh/pixel space, ta dùng proxy:

$$\text{score}_i \approx \sum_{t \in \mathcal{T}(i)} w_{i,t}\; E_{\text{tile}}[t]$$

* [ ]  **Kernel input**:
    `means2D_px (N×2)`, `radii (N)`, `E_tile (H_t×W_t)`, `T=16`, `visible mask`.
* [ ]  **Kernel output**:
    `fw_score (N)` hoặc (N×1) để drop-in vào pipeline.
* [ ]  **Chọn weight w_{i,t}**:
    * mặc định: `w=1` (rất dễ implement, surprisingly works baseline-wise)
    * tốt hơn chút: `w = exp(-||c_t - μ_i||^2 / (2σ_i^2))` với `σ_i≈r_i/3`
    * hoặc normalize theo số tiles để khỏi ưu ái Gaussians rất to.
* [ ]  **Implementation plan A (no custom CUDA)**: “mean-tile gather”
    * tile_id_i = floor(x_i/T) + W_t*floor(y_i/T)
    * score_i = E_tile_flat[tile_id_i] * g(r_i)
        **Ưu:** O(N), vectorized, không loop.
        **Nhược:** bỏ qua multi-tile overlap (thô).
* [ ]  **Implementation plan B (CUDA ker

    Splat Regression Models

    - mỗi thread xử lý 1 Gaussian: compute tx0..tx1,ty0..ty1; loop tiles (thường rất ít); sum E_tile.
    * Complexity ~ O(∑ tiles_i). Với tile=16 và ellipse radius vừa phải, tiles_i thường 1–9.
        **Ưu:** gần đúng tile overlap như rasterizer; không cần build list.
        **Nhược:** cần viết kernel.

**Quan trọng về autograd:**

* [ ]  Toàn bộ score computation đặt trong `torch.no_grad()` và dùng `.detach()` từ các tensor (image.grad cũng detach).
* [ ]  Không dùng `.item()`, `.cpu()`, `.numpy()` trong loop (sync GPU gây chậm và có thể làm bạn tưởng “code treo”).

---

### 5) Data structures cần thêm trong `GaussianModel` để “implementation-consistent”

Baseline densification của 3DGS tích luỹ thống kê:

* `xyz_gradient_accum` và `denom` được update bởi `add_densification_stats(...)`
* `densify_and_prune(...)` dùng `grads = xyz_gradient_accum / denom` để quyết định clone/split/prune [GitHub+1](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)

Để tích hợp FW-score mà ít phá code:

#### 5A) Thêm buffer song song (khuyến nghị)

* [ ]  Trong `GaussianModel.__init__/training_setup`:
    * `self.fw_score_accum = zeros([N,1], device=cuda)`
    * `self.fw_denom = zeros([N,1], device=cuda)`
* [ ]  Thêm method `add_fw_stats(fw_score, visibility_filter)`:
    * update_filter = visibility_filter (indices) hoặc bool mask
    * `fw_score_accum[update_filter] += fw_score[update_filter].unsqueeze(-1)`
    * `fw_denom[update_filter] += 1`
* [ ]  Trong `densification_postfix(...)`: reset 2 buffer này giống cách reset `xyz_gradient_accum/denom/max_radii2D`. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py)
* [ ]  Trong `prune_points(mask)`: cũng prune các buffer này theo mask (giống prune xyz_gradient_accum/denom). [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py)

#### 5B) Không muốn sửa nhiều: nhét fw_score vào xyz_gradient_accum

* [ ]  Hack nhanh: set `xyz_gradient_accum = fw_score_accum` và `denom = fw_denom`.
* [ ]  Nhưng dễ gây “confusing semantics” khi debug/ablation → paper khó thuyết phục hơn.

---

### 6) Gắn vào `train.py` đúng vị trí (không phá schedule hiện tại)

Dựa trên loop trong `train.py`, densification block nằm trong `torch.no_grad()` sau `loss.backward()` và trước optimizer step. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)

**Checklist chính xác theo thứ tự thời gian:**

* [ ]  (Trước backward) Nếu iteration nằm trong cửa sổ densify:
    * `if iter < densify_until && iter % densification_interval == 0: image.retain_grad()`
* [ ]  `loss.backward()`
* [ ]  `with torch.no_grad():`
    1. build residual stats:
        * `residual_img = image.grad.detach()` (hoặc proxy)
        * pool → `E_tile`
    2. compute `means2D_px`:
        * _không dùng viewspace_points values_ [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)
        * tự project từ `gaussians.get_xyz` + camera matrices
    3. compute `fw_score` (Plan A or B)
    4. `gaussians.add_fw_stats(fw_score, visibility_filter)`
    5. tại densify interval, gọi densify:
        * sửa `gaussians.densify_and_prune(...)` để dùng `fw_score_accum/fw_denom` (hoặc combine với baseline grads)
* [ ]  Sau densify/prune: `optimizer.step()` và `optimizer.zero_grad(set_to_none=True)` như baseline. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)

---

### 7) “Không phá backprop graph” — các bẫy thường gặp

* [ ]  **Không được** dùng residual map “có graph” trong no_grad block.
    Nếu bạn dùng proxy residual `abs(image - gt)` mà quên `.detach()`, bạn sẽ giữ graph lâu hơn cần thiết và dễ leak VRAM.
* [ ]  **Không gọi** `torch.autograd.grad(loss, image, ...)` mỗi densify step nếu bạn vẫn gọi `loss.backward()` (sẽ double backward cost).
    Cách sạch là `image.retain_grad()` trước backward rồi lấy `image.grad` sau backward.
* [ ]  **Không assume** `viewspace_points` chứa projected means2D. Nó là “dummy zero tensor” để lấy gradient. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)

---

### 8) Performance guardrails (để không tăng wall-clock quá nhiều)

* [ ]  Chỉ compute `image.grad` và tile residual **khi đến densification step** (vd mỗi 100 iters).
* [ ]  Tile pooling là O(HW) nhưng chạy bằng CUDA kernel của PyTorch; thường rẻ hơn nhiều so với rasterization/backward.
* [ ]  Score kernel nên chạy O(N) hoặc O(N·k_tiles) với k_tiles nhỏ; tránh bất kỳ Python-loop qua N Gaussians.
* [ ]  Nếu project means2D bằng PyTorch, nhớ `.detach()` và làm trong no_grad; tránh tạo graph vô ích.
* [ ]  Nếu muốn ultimate speed: patch rasterizer để expose tileRanges/point_list (đã có sẵn trong forward theo paper) rồi làm reduce. [ar5iv](https://ar5iv.org/pdf/2308.04079)

---

### 9) Debug/Validation checklist (để biết mình “đúng hướng”)

* [ ]  Sanity 1: `image.grad` không None ở densify iteration (đã retain_grad).
* [ ]  Sanity 2: distribution của projected means2D cho visible gaussians nằm trong ảnh (đa số).
* [ ]  Sanity 3: `fw_score` > 0 cho vùng residual lớn; và top-K gaussians được chọn nằm ở vùng đó khi visualize (project ra 2D).
* [ ]  Sanity 4: Sau densify, loss không explode; số gaussians tăng đúng expected; prune vẫn chạy.
* [ ]  Sanity 5: wall-clock tăng không đáng kể (mục tiêu: +<5–10% so với baseline; nếu hơn, thường do projection sai shape gây sync, hoặc Python-loop).

---

## Một “sell point” rất quan trọng cho reviewer (gắn trực tiếp với theorem & implementation)

* **Theorem 1 của SRM** nói update theo v/A/b của splat measure phụ thuộc vào “residual field” δF được “lọc” qua density ρA,b.
* Trong 3DGS, rasterizer đã tổ chức computation theo tile (16×16) và duplicate splats theo tile để xử lý nhanh. [ar5iv](https://ar5iv.org/pdf/2308.04079)
* Ý tưởng FW-densification của mình (ở level implementation) chính là:
    “Thay vì densify theo heuristic gradient-norm của means2D, ta densify theo **correlation δF với footprint 2D của Gaussian**, được xấp xỉ hiệu quả bằng tile pooling + overlap sum.”
    → Đây là điểm nối “theory-based” ↔ “codepath hiện hữu” khá thuyết phục nếu bạn làm rõ pipeline như checklist trên.

---

Nếu bạn muốn bước tiếp theo hợp lý (không hỏi thêm), mình sẽ draft luôn **một pseudo-code block “FWScoreDensifyStep()”** đúng theo các bước trên (Plan A vectorized trước, rồi Plan B CUDA-kernel), và chỉ ra chính xác bạn cần sửa 3 file nào + signature thay đổi ra sao để merge clean vào repo.

Splat Regression Models

* * *

**You:**

tôi chọn theo hướng plan b cuda-kernel. phân tích lại pseudo-code block sau đó hãy verify lại implementation bạn để xuất có đang theo đúng theory hay không, kiểm tra chính xác từng bước và đảm bảo tôi không đang đi sai hướng. Verify lại đảm bảo không hallucinate, và chính xác với pipeline của 3dgs

* * *

**ChatGPT:**

Dưới đây là một “review + verification” rất sát code/pipeline 3DGS (graphdeco-inria) cho **Plan B: CUDA-kernel** theo hướng _Conditional Gradient / Sliding Frank–Wolfe densification_ mà bạn chọn. Mình sẽ làm 4 việc:

1. **Chốt lại pipeline densification gốc của 3DGS** (để không lệch implementation).
2. **Formalize “FW-style score”** sao cho _bám thẳng Theorem 1 / Example 1_ trong _Splat Regression Models_ và chỉ rõ chỗ nào là xấp xỉ.
3. **Draft một section “Method” paper-ready**: objective, proxy LMO, thuật toán, complexity.
4. **Implementation Notes dạng checklist + pseudo-code CUDA-kernel** (mapping đúng training loop), và “theory ↔ implementation consistency check” từng bước.

---

## 1) Pipeline densification gốc của 3DGS đang làm gì (để ta không “lệch ray”)

### 1.1 `viewspace_points` trong 3DGS thực ra là “dummy tensor” để hứng gradient 2D

Trong `gaussian_renderer/__init__.py`, repo gốc tạo một tensor `screenspace_points` toàn 0, bật `requires_grad=True`, rồi pass nó vào rasterizer như `means2D`. Comment trong code nói rõ mục đích là **để PyTorch trả gradient theo mean 2D**. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)

=> Điều này xác nhận: **giá trị của `viewspace_points` không phải tọa độ 2D thực**, nó chỉ là “nơi chứa grad”.

### 1.2 Densification stats được cập nhật sau `loss.backward()` trong `train.py`

Trong training loop, sau khi render + compute loss + `loss.backward()`, code vào block `with torch.no_grad()` và làm:

* update `max_radii2D[visibility_filter]`
* `add_densification_stats(viewspace_point_tensor, visibility_filter)`
* theo chu kỳ thì `densify_and_prune(...)` [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)

=> Nếu bạn muốn thêm score mới, chỗ “đúng pipeline” để compute **là sau backward, trong no_grad**, giống hệt nơi họ accumulate stats.

### 1.3 Heuristic densification gốc = norm(grad của means2D)

Trong `GaussianModel.add_densification_stats`, họ lấy `viewspace_point_tensor.grad[update_filter, :2]`, lấy norm rồi cộng vào `xyz_gradient_accum` và tăng `denom`. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py)

=> Densification gốc chọn “điểm cần densify” dựa trên **độ lớn gradient w.r.t. mean2D** (không trực tiếp nhìn vào residual / error map).

**Kết luận “ground truth”**: Plan B của bạn (score kernel) phải _tương thích với chỗ này_:

* chạy sau backward (để có error signal),
* dùng `visibility_filter` như gating,
* accumulate vào buffer kiểu `*_accum` và `denom`,
* rồi thay criterion trong `densify_and_prune()`.

---

## 2) Theory: FW-score phải bám Theorem 1 của SRM như thế nào?

### 2.1 Theorem 1 + Example 1 của SRM cho ta “đúng công thức” gradient kiểu splat

Trong SRM, họ định nghĩa splat model $f_\mu(x) = \mathbb{E}[v\,\rho_{A,b}(x)]$.
Theorem 1 nói gradient (Wasserstein part) theo $v$ là:

$$\nabla_v F(f_\mu)(v,A,b) \;=\; \int \delta F[f_\mu](x)\,\rho_{A,b}(x)\,\pi(dx)$$

Và Example 1 nhấn mạnh $\delta F$ chính là **error signal vector được backprop** trong pipeline deep learning.

Splat Regression Models

ng**: nếu mình làm một “FW-style” LMO/score để chọn atom/splat nào cần thêm (densify), thì hạt nhân chính là **tương quan giữa error signal $\del:contentReference[oaicite:7]{index=7}(\rho$**.

### 2.2 Mapping sang 3DGS: $\delta F$ trong vision nên lấy ở đâu?

Trong 3DGS, output trực tiếp là **ả

Splat Regression Models

u loss là $ \mathcal{L}(I, I^{gt})$ thì:

* “đúng-theory” nhất là dùng: $$\delta F(u) \equiv \frac{\partial \mathcal{L}}{\partial I(u)}$$ với $u$ là pixel coordinate.

Trong thực tế PyTorch: lấy **`image.grad`** sau backward. Nhưng phải nhớ `image` không phải leaf tensor nên bạn cần `image.retain_grad()` trước backward để `image.grad` tồn tại.

Điểm này bám thẳng Example 1: “error signal vector propagated through backprop” = $\delta F$.

### 2.3 Footprint $\rho$ trong 3DGS có thật sự là “splat kernel” không?

Có. Trong diff-gaussian-rasterization forward pass:

* họ project 3D point → NDC → pixel bằng `ndc2Pix`,
* tính radius $r = \lceil 3\sqrt{\lambda_{\max}}\rceil$,
* và render per-pixel bằng công thức dạng \alpha = \text:contentReference[oaicite:10]{index=10}xt{power}} với `power = -0.5 * (quadratic form)`. [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)

`ndc2Pix` và `getRect` (bbox theo tile) được định nghĩa rõ:

$$\text{ndc2Pix}(v,S)=((v+1)S-1)\cdot 0.5$$

và `getRect` dùng `BLOCK_X/BLOCK_Y` để ra bbox tile. [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)

=> Nói cách khác: 3DGS có đúng “splat kernel” trong image plane; và rasterizer đã có cơ chế tile-binning để xử lý footprint.

**Nhưng**: footprint “thực” của 3DGS còn bị nhân bởi transmittance $T$ do alpha compositing (renderCUDA dùng biến `T` cập nhật mỗi gaussian). [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)
Plan B của bạn khi chỉ dùng bbox/tile-weight sẽ là **xấp xỉ** của kernel hiệu dụng (vì bỏ qua $T$ và bỏ qua conic anisotropic đầy đủ).

Ta cần thừa nhận chỗ xấp xỉ này trong paper, và đưa “gating/mitigation” hợp lý.

---

## 3) Plan B CUDA-kernel: Formalize “FW-score” để vừa theory-based vừa implement được

Mục tiêu của densification trong FW view: **chọn gaussian nào “có tiềm năng giảm loss lớn nhất nếu ta tăng capacity quanh nó”**.
Vì 3DGS densify bằng _split/clone_ các gaussians hiện có, ta làm **restricted LMO**: chọn trong tập ứng viên là các gaussians đang có (thay vì search toàn không gian).

### 3.1 Proxy objective để reviewer không bắt bẻ

Để có “proxy convex” hợp lý (và reviewer dễ nuốt):

* Với camera cố định và geometry cố định tại iteration hiện tại, ta coi mapping từ “các atom” tới ảnh là “xấp xỉ tuyến tính” quanh current point (first-order Taylor).
* Trong proxy đó, việc “thêm một atom mới” tương đương thêm một basis function footprint lên ảnh.
* Conditional gradient/Frank–Wolfe chọn atom bằng cách **tối ưu inner product với gradient**.

Điểm tựa lý thuyết:

* Theorem 1 cho gradient dạng tích phân của $\delta F$ với $\rho$.
* Example 1 cho ta cách lấy $\delta F$ từ backprop.
* (Ngoài SRM) FW/CGM là chuẩn trong convex optimization. Jaggi 2013 là reference “Frank-Wolfe revisited”. [Nick Boyd](https://nboyd.github.io/adcg.pdf)
* ADCG (Alternating Descent Conditional Gradient) chính là “FW + local search (sliding)” trên bài toán sparse inverse problems. Đây cực hợp với “densify + SGD refine” của 3DGS. [arXiv](https://arxiv.org/abs/1507.01562)
* “Sliding Frank-Wolfe” cũng tồn tại như một biến thể cho sparse measures. [EPFL Biomedical Imaging Group](https://bigwww.epfl.ch/publications/denoyelle1901.pdf)

### 3.2 Định nghĩa score “implementation-consistent” với Theorem 1

Ta muốn một score per-gaussian $s_i$ sao cho:

$$s_i \approx \left\| \int \frac{\partial \mathcal{L}}{\partial I(u)} \; \rho_i(u)\; du \right\|$$

Trong SRM, đây chính là

Splat Regression Models

) (gradient theo amplitude vector).

Splat Regression Models

delta F(u)\in \mathbb{R}^3). Vậy $s_i$ là norm của vector tích phân.

**Tile approximation** (để CUDA chạy nhanh):
chia ảnh thành tiles (đúng như rasterizer). Paper 3DGS mô tả renderer chia screen thành tiles 16×16 và duplicate gaussians vào tile list. [ar5iv](https://ar5iv.org/pdf/2308.04079)
Rasterizer implementation dùng `getRect` để ra bbox tile từ `(point_image, radius)`. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)

* Tính $\mathbf{G}_t = \sum_{u\in t} \delta F(u)$ cho mỗi tile $t$.
* Với gaussian $i$, bbox tile là $ \mathcal{T}_i$.
* Xấp xỉ tích phân bằng tổng: $$\hat{\mathbf{g}}_i = \sum_{t\in\mathcal{T}_i} w_{i,t} \mathbf{G}_t
    \quad\Rightarrow\quad
    s_i = \|\hat{\mathbf{g}}_i\|_2$$ với $w_{i,t}$ là weight xấp xỉ $\int_{u\in t}\rho_i(u)\,du$.
    Implementation nhẹ nhất: $w_{i,t}=1$.
    Implementation “đỡ bịa” hơn: dùng

    Splat Regression Models

    eo khoảng cách từ tile center tới mean2D, với $\sigma \approx r_i/3$ vì rasterizer radius dùng $3\sqrt{\lambda_{\max}}$. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)

**Điểm mạnh**: cách này bám đúng cấu trúc Theorem 1: “$\delta F$ * kernel footprint” → integrate/sum.

---

## 4) “Method” paper-ready draft (ngắn nhưng đủ formal)

### Method: Conditional-Gradient Densification via Tile-Integrated Image Gradients

#### Objective

Given a set of 3D Gaussians $\{ \theta_i \}_{i=1}^N$ with parameters $\theta_i = (v_i, A_i, b_i, \dots)$, the renderer produces an image $I = \mathcal{R}(\{\theta_i\}; c)$ for camera $c$. We minimize a photometric loss:

$$\min_{\{\theta_i\}} \; \mathcal{L}\big(\mathcal{R}(\{\theta_i\}; c), I^{gt}\big)$$

Densification adds new Gaussians (via split/clone) to increase representational capacity.

#### Proxy linear minimization (restricted LMO)

Let $\delta I(u) := \frac{\partial \mathcal{L}}{\partial I(u)}$ be the image-space error signal (first variation) at pixel $u$. Under the splat-model calculus (Theorem 1 in SRM), the gradient w.r.t. an atom’s amplitude $v_i$ takes the form:

$$\nabla_{v_i}\mathcal{L} \propto \int \delta I(u)\,\rho_i(u)\,du$$

We define a **restricted linear minimization oracle** over existing atoms:

$$i^\star \in \arg\max_{i\in\{1,\dots,N\}} \left\| \int \delta I(u)\,\rho_i(u)\,du \right\|$$

and densify aro

Splat Regression Models

t/clone), followed by standard SGD updates (“sliding/local refinement”).

#### Tile-based approximation

The renderer uses tile-binning; we mirror this discretization. Let tiles $t$ be $B\times B$ pixel blocks (default $B=16$). For each tile:

$$\mathbf{G}_t := \sum_{u\in t} \delta I(u)\in\mathbb{R}^3$$

For each Gaussian, we compute its tile rectangle $\mathcal{T}_i$ using the same NDC-to-pixel and rectangle rules as the rasterizer (`ndc2Pix`, `getRect`). [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
We then approximate:

$$\hat{\mathbf{g}}_i := \sum_{t\in\mathcal{T}_i} w_{i,t}\mathbf{G}_t, \quad s_i := \|\hat{\mathbf{g}}_i\|_2$$

We densify the top-$K$ Gaussians by score $s_i$.

#### Algorithm

* Forward render $I$.
* Backward compute $\delta I = \partial\mathcal{L}/\partial I$.
* CUDA-kernel (1) reduces $\delta I$ into per-tile sums $\mathbf{G}_t$.
* CUDA-kernel (2) computes $s_i$ for all Gaussians using rasterizer-consistent tiling.
* Accumulate scores over iterations, then densify periodically.

#### Complexity

Let $H,W$ be image resolution; number of tiles $T = \lceil H/B\rceil\lceil W/B\rceil$.

* Tile reduction: $O(HW)$.
* Gaussian scoring: $O(\sum_i |\mathcal{T}_i|)$, where $|\mathcal{T}_i|$ is number of tiles touched; typically small because bounding radius is limited. (The rasterizer already uses the same rectangle for work assignment.) [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)

---

## 5) “Theory” proxy section (liên hệ trực tiếp Theorem 1 + tính đúng/đủ)

### 5.1 What is **exactly** justified by SRM Theorem 1?

Theorem 1 states that for splat models $f_\mu(x)=\mathbb{E}[v\rho_{A,b}(x)]$, the Wasserstein gradient w.r.t. $v$ is an integral of the first variation against the splat kernel:

$$\nabla_v F(f_\mu)(v,A,b)=\int \delta F(x)\rho_{A,b}(x)\pi(dx)$$

Example 1 states $\delta F$ equals the backpropagated error signal (gradient of loss w.r.t. model output).

=> Vì vậy, nếu ta xem **ảnh render** là output function trên domain pixel, thì việc lấy $\delta I(u)=\partial\mathcal{L}/\partial I(u)$ và correlate với footprint của atom là **đúng dạng** mà theorem dự đoán.

### 5.2 Phần nào là “proxy/xấp xỉ” khi áp dụng vào 3DGS?

3DGS renderer không phải simple sum $ \sum v_i\rho_i(u)$; nó là alpha compositing có transmittance $T$ và culling, thresholding, v.v. [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)
Vì vậy:

* Score $\sum \delta I(u)\rho_i(u)$ là **đúng chính xác** với “splat regression” tuyến tính.
* Với 3DGS, đó là **proxy** cho gradient theo $v$ của _effective kernel_ (đã nhân $T$).
    Nếu ta bỏ $T$, score có thể over-estimate các Gaussians bị che khuất.

Reviewer sẽ hỏi chỗ này. Câu trả lời chuẩn là:

* Ta dùng score như **selection heuristic** cho densification, không claim hội tụ toàn cục.
* Ta “implementation-consistent” ở chỗ ta dùng **cùng tile rectangle** như rasterizer (ndc2Pix/getRect/radius), và dùng **backprop error signal** đúng như Example 1. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
* Optional: ta thêm **visibility/sensitivity gating** để giảm bias do occlusion (mình đề xuất dưới).

### 5.3 Sliding Frank–Wolfe / ADCG connection có “bịa” không?

Không bịa: có dòng nghiên cứu rõ về “conditional gradient for sparse in

Splat Regression Models

lternating descent conditional gradient” (ADCG), và sliding FW variants. [arXiv+2EPFL Biomedical Imaging Group+2](https://arxiv.org/abs/1507.01562)

Splat Regression Models

ới 3DGS:

* **Global step** (CG): chọn atom để thêm (densify).
* **Local step** (sliding): tối ưu continuous parameters bằng SGD/Adam sau khi thêm atom (3DGS vốn làm vậy).

Cái bạn làm là một biến thể “restricted oracle” (chỉ densify quanh atoms có sẵn), nhưng reviewer thường chấp nhận miễn bạn nói rõ đây là restriction để giữ hiệu năng và reuse pipeline.

---

## 6) Pseudo-code Plan B (CUDA-kernel) + verify từng bước theo theory và theo pipeline 3DGS

Mình sẽ đưa pseudo-code “cứng” và check từng dòng xem có khớp pipeline không.

### 6.1 Pseudo-code (Python-level) — chỗ gắn vào train loop

**Ở ngay sau render và trước backward:**

```python
image = render_pkg["render"]
if viewpoint_cam.alpha_mask is not None:
    image = image * viewpoint_cam.alpha_mask
image.retain_grad()   # IMPORTANT: to access image.grad after backward
loss = ...
loss.backward()
```

**Trong `with torch.no_grad()` block, nơi hiện đang gọi `add_densification_stats`:**

```python
with torch.no_grad():
    g_img = image.grad                     # (3,H,W) = δI(u)
    vis = visibility_filter                # bool[N]
    radii = render_pkg["radii"]            # (N) pixel radius
    xyz  = gaussians.get_xyz               # (N,3)
    proj = viewpoint_cam.full_proj_transform  # same one rasterizer uses

    tile_G = tile_reduce_sum3_cuda(g_img, B=16)   # (T_y,T_x,3)
    score  = fw_score_cuda(xyz, radii, vis, proj, tile_G, W, H, B=16)
    gaussians.fw_score_accum[vis] += score[vis]
    gaussians.fw_score_denom[vis] += 1
```

**Trong densify step:**

```python
scores = gaussians.fw_score_accum / gaussians.fw_score_denom.clamp_min(1)
gaussians.densify_and_prune_using_scores(scores, ...)
```

#### Verification theo pipeline 3DGS

* `visibility_filter` đúng là thứ họ dùng để update stats và radii. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)
* `image.retain_grad()` là cần thiết vì otherwise `image.grad` sẽ không tồn tại (image không leaf). Đây là implementation detail nhưng hoàn toàn “graph-safe” nếu chỉ retain gradient.
* Chạy trong `no_grad` là đúng pattern của repo. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)

### 6.2 Pseudo-code CUDA-kernel 1: Tile reduction

Input: `g_img` shape `[3,H,W]` float32.
Output: `tile_G` shape `[Ty,Tx,3]` float32.

Mỗi tile $16\times16$:

$$\mathbf{G}_t = \sum_{u\in tile} \delta I(u)$$

(đây là Riemann-sum coarse cho $\int \delta I(u)du$).

**Theory check**: $\delta I$ chính là $\delta F$ trong Example 1 (error signal backprop).
**Implementation check**: kernel này không cần knowledge về gaussians, chỉ reduce.

### 6.3 Pseudo-code CUDA-kernel 2: Compute FW-score per Gaussian (rasterizer-consistent)

Key: dùng **cùng công thức projection + ndc2Pix + getRect** như rasterizer.

Rasterizer preprocess làm:

* `p_hom = transformPoint4x4(p_orig, projmatrix)`
* `p_proj = p_hom.xyz / p_hom.w`
* `point_image = { ndc2Pix(p_proj.x,W), ndc2Pix(p_proj.y,H) }`
* `my_radius = ceil(3*sqrt(lambda_max))`
* `getRect(point_image,my_radius,rect_min,rect_max,grid)` [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)

**Bạn nên y hệt**.

**Score kernel** (cho gaussian i):

1. Nếu `vis[i]==false` → `score[i]=0`.
2. Compute `point_image` bằng `projmatrix` và `ndc2Pix`. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)
3. Dùng `getRect(point_image, radius[i])` ra `rect_min, rect_max` (note: max là exclusive). [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
4. Accumulate:

$$\hat{\mathbf{g}}_i = \sum_{t\in [rect\_min,rect\_max)} w_{i,t}\,\mathbf{G}_t,\quad s_i=\|\hat{\mathbf{g}}_i\|$$

**Nếu bạn muốn weight “đỡ bịa”**:

* set $\sigma = r_i/3$
* $w_{i,t}=\exp(-\tfrac12 \|c_t-\mu_i\|^2/\sigma^2)$ với $c_t$ tile center.

**Theory check**: Đây là xấp xỉ tile-level cho tích phân $\int \delta I(u)\rho_i(u)\,du$, bám Theorem 1.
**Implementation check**: Bbox theo tile được tính đúng theo rasterizer util `getRect` và `ndc2Pix`. [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)

---

## 7) “Implementation-consistent” checklist (mapping đúng vào gaussian-splatting training loop + data structures + không phá graph)

### A. Lấy residual / error signal ở đâu?

* [ ]  **Không dùng** `image.detach() - gt` nếu bạn muốn “theory clean”.
* [ ]  Dùng **`g_img = image.grad`** sau backward vì đó chính là $\partial \mathcal{L}/\partial I$ (error signal) như Example 1 nói.
* [ ]  Phải gọi `image.retain_grad()` trước backward.

### B. Tile-binning score kernel ra sao đ

Splat Regression Models

] Tile size phải khớp `BLOCK_X/BLOCK_Y` của rasterizer (paper nói 16×16). [ar5iv](https://ar5iv.org/pdf/2308.04079)

* [ ]  Mapping NDC→pixel dùng đúng: $$\text{ndc2Pix}(v,S)=((v+1)S-1)/2$$ [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
* [ ]  Bounding rectangle dùng đúng `getRect` (rect_max exclusive). [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
* [ ]  Dùng `projmatrix` y hệt rasterizer preprocess (full_proj_transform pass vào raster_settings). [GitHub+1](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/gaussian_renderer/__init__.py)

### C. Data structures cần thêm gì trong GaussianModel?

Thêm các buffer giống pattern `xyz_gradient_accum/denom`:

* [ ]  `fw_score_accum: (N,) float32`
* [ ]  `fw_score_denom: (N,) float32/int32`
* [ ]  reset khi densify/prune (giống cách họ reset denom/accum sau densify).

### D. Không phá backprop graph

* [ ]  Compute score trong `with torch.no_grad()` đúng như repo đang làm cho densification. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)
* [ ]  CUDA kernels chỉ đọc `image.grad`, `xyz`, `radii`, `projmatrix` và ghi ra score tensor mới; không backprop qua score (score chỉ là stats).

### E. Không tăng wall-clock quá nhiều

* [ ]  Tile reduction: O(HW) nhỏ.
* [ ]  Score kernel: O(∑ tiles_touched). radius/rect được thiết kế để tiles_touched thường nhỏ; chính rasterizer cũng dựa vào đó để chạy nhanh. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)
* [ ]  Tối ưu: chỉ compute score ở các iteration nằm trong “densification window” và/hoặc đúng densification_interval, thay vì mọi iter.

### F. Debugging sanity checks (rất quan trọng để tránh “theory hay nhưng code sai”)

* [ ]  **Check projection agreement**: với vài gaussians, log `point_image` kernel bạn tính vs (nếu bạn tạm patch rasterizer xuất `points_xy_image`) hoặc bằng cách render marker.
* [ ]  **Check bbox agr

    Splat Regression Models

    x kernel bạn tính phải match logic getRect. [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
* [ ]  **Check score correlation**: score_i của bạn nên tương quan với norm gradient theo SH/DC của gaussian (vì đều đo “ảnh hưởng lên loss”). Nếu tương quan ~0 → bug mapping coordinate/tile.

---

## 8) Reviewer-mode: phương pháp này “tốt hơn” gì? Sell point là gì? Và rủi ro thật sự nằm ở đâu?

### Sell point mạnh nhất (đúng kiểu reviewer thích)

1. **Theory-grounded**:
    * Bạn không “bịa score”: score xuất phát trực tiếp từ Theorem 1: gradient theo amplitude = integral of error signal × kernel.
    * Error signal lấy từ backprop đúng như Example 1.

        Splat Regression Models

        lementation-consistent với renderer**:
    * Bạn không invent tiling; bạn reuse đúng `ndc2Pix/getRect` semantics của rasterizer, nên bbox/tile overlap khớp “thứ rasterizer thực sự render”. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
2. **Giải quyết một gap thật trong 3DGS densification**:
    Densification gốc dựa vào norm(grad mean2D) — nó là heuristic khó diễn giải và có thể “densify vì sensitivity” hơn là “densify vì error”. [GitHub+1](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py)
    Phương pháp của bạn thay bằng criterion trực tiếp gắn vào $\partial \mathcal{L}/\partial I$ (pixel error signal), nên dễ justify và dễ debug.
3. **Chi phí rẻ + dễ plug-in**:
    Vì bạn làm tile-reduction + bbox scan, overhead có thể nhỏ và densification vốn không chạy mỗi iter.

### Rủi ro thật (để tránh stuck)

Rủi ro #1: **Occlusion / transmittance mismatch**

* Lý thuyết SRM dùng kernel $\rho$ “đơn”; 3DGS effective kernel bị nhân bởi transmittance $T$. [Hugging Face](https://huggingface.co/spaces/ashawkey/LGM/blob/620664261dcd84dfdf1adb458e4b55ab279b4d25/diff-gaussian-rasterization/cuda_rasterizer/forward.cu)
* Nếu score chỉ dựa trên bbox overlap (không biết $T$), có thể pick gaussian “nằm sau” bề mặt nhưng bbox vẫn phủ tile error.

**Mitigation thực dụng, ít phá code**:

* Gating bằng `visibility_filter` (ít nhất đảm bảo gaussian có radius>0 và được rasterizer consider). [GitHub+1](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/train.py)
* Thêm “sensitivity gate” bằng grad norm theo means2D (cái 3DGS đã tính). Nếu gaussian bị occluded, grad thường ~0. [GitHub](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/scene/gaussian_model.py)
    Cụ thể: `final_score_i = s_i * clamp(grad2D_norm_i, 0, τ)` hoặc `final_score_i = s_i` nếu grad2D_norm_i>ε.

Rủi ro #2: **Score đo gradient theo v (mà densify lại split b/A)**

* Theorem 1 formula bạn bám nhất là gradient theo $v$.
* Nhưng densification split/clone đang can thiệp geometry (b, A) nhiều hơn weight v.

Reviewer có thể hỏi: “Bạn chọn theo ∇v, sao lại split geometry?”

Câu trả lời ổn (và đúng thực tế):

* Trong splat models, khi region có error lớn, gradient theo v thường lớn vì thiếu capacity; split/clone tăng basis functions ở region đó → cho SGD có thể điều chỉnh b/A/v tốt hơn. Đây là “restricted oracle + local refinement” kiểu ADCG: global step chọn nơi cần thêm atom, local step optimize continuous params. [arXiv+1](https://arxiv.org/abs/1507.01562)

---

## 9) Chốt: Plan B của bạn có đang “đi đúng hướng” không?

**Mình đánh giá (theo reviewer criteria) là: hướng này đủ “theory-based + implementable + relevant” để theo, với điều kiện bạn làm 3 thứ để tránh bị bắt bẻ:**

1. **Score phải dùng $\delta I = \partial \mathcal{L}/\partial I$** (tức `image.grad`), không chỉ `image-gt`. Khi đó bạn claim “theory-consistent with SRM Example 1/Theorem 1” rất sạch.
2. **BBox/tile mapping phải clone đúng rasterizer** (ndc2Pix + getRect semantics). Bạn đã có công thức chuẩn từ auxiliary.h và preprocess forward.cu. [Hugging Face+1](https://huggingface.co/spaces/ashawkey/LGM/blob/66cb594ac5415bf74e12be85594f648e1bd39e7c/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h)
3. **Phải xử lý occlusion mismatch** bằng một gating hợp lý (ít nhất visibility_filter; tốt hơn là nhân thêm sensitivity gate từ grad2D). Nếu không, bạn có nguy cơ debug địa ngục: score “chỉ vào chỗ lỗi” nhưng lại densify sai gaussian (nằm sau mặt).

Nếu 3 điểm này được làm, thì paper story sẽ rất “sellable”:

* “We replace heuristic

    Splat Regression Models

    conditional-gradient-inspired, SRM-theorem-grounded crite

    Splat Regression Models

    image-space error signals with splat footprints, implemented efficiently via rasterizer-consistent tile reductions.”

Không màu mè, không bịa theorem, và code chạy được.

---

Nếu bạn muốn bước tiếp “không bị stuck”, hướng hợp lý nhất là bạn implement kernel theo **đúng ndc2Pix/getRect** trước, rồi làm **sanity correlation test** với `features.grad` hoặc `viewspace_points.grad` để bắt bug coordinate sớm. Khi correlation bắt đầu “có lý”, mới tune weight $w_{i,t}$ và gating.

Splat Regression Models

* * *

**You:**

Làm sao để experiment và kiểm tra method của tôi tốt hơn ? Tôi cần bạn nghĩ cách thiết kế experiment để verify dựa trên theory này và bạn nghĩ phương pháp của tôi có thể tốt hơn những phương pháp khác ở điểm nào và metrics và giải thích rõ ràng tại sao ? Hãy phân tích chi tiết thí nghiệm và cách verify dựa trên những điểm tốt của phương pháp ta đang phân tích.

* * *

**ChatGPT:**

Để “prove” phương pháp densification kiểu **FW / tile‑integrated image‑gradient** của bạn _tốt hơn_ (và quan trọng hơn: **đúng-theory, không phải chỉ thay heuristic**) thì nên thiết kế thí nghiệm theo 3 tầng:

1. **Tầng sanity / debug**: chứng minh code đang đo đúng thứ theo SRM Theorem 1.
2. **Tầng “theory verification”**: chứng minh score của bạn là một **oracle chọn densify tốt** (tương quan với loss decrease / gradient thật).
3. **Tầng benchmark**: PSNR/SSIM/LPIPS + hiệu năng + budget‑quality tradeoff trên các dataset chuẩn.

Mình sẽ đi rất cụ thể từng thí nghiệm, metrics, và “tại sao” (gắn thẳng vào điểm mạnh của method bạn).

---

## 0) “Claim” bạn đang bán cho reviewer là gì?

**SRM (Splat Regression Models)** cho bạn một statement cực mạnh:

* **Example 1** nói $\delta F$ chính là _error signal backprop_ trong pipeline deep learning.
* **Theorem 1** cho gradient theo “amplitude/output vector” $v$: $$\nabla_{v_i} F \;=\; \int \delta F(x)\,\rho_i(x)\,d\pi(x)$$

Nói tiếng 3DGS: _gradient theo “màu/SH” của một Gaussian_ là “tích phân” của **image‑space error signal** qua footprint của Gaussian.

Splat Regression Models

Phương pháp của bạn (Plan B CUDA‑kernel) đúng bản chất là:

* Lấy $\delta I(u)=\partial\mathcal{L}/\partial I(u)$ (từ `image.grad`)
* Tích phân (xấp xỉ bằng tile sum) trên footprint $\rho_i$
* Lấy norm để ra **score** chọn densify.

=> Vì thế “điểm mạnh” của bạn so với ADC gốc (dựa vào grad mean2D) là:
**densification criterion = proxy của $\|\nabla_{v_i} F\|$**, tức là chọn đúng nơi “nếu tăng capacity ở đây thì loss có cơ hội giảm nhiều”.

---

## 1) Bạn có thể tốt hơn ai, và ở điểm nào?

Đặt baseline rõ ràng (vì reviewer sẽ hỏi “so với ai?”):

### (A) 3DGS gốc (Adaptive Density Control)

3DGS gốc dùng interleaved optimization/density control + renderer visibility‑aware và nhắm real‑time (≥30 fps 1080p). [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)
Nhưng densify criterion truyền thống dựa nhiều vào grad view‑space (mean2D), dễ bị lệch “error‑driven”.

**Kỳ vọng lợi thế của bạn**:

* densify “đúng chỗ sai” (high error & overlapping footprint), không chỉ “nhạy” theo 2D mean.

### (B) Revising Densification in Gaussian Splatting (ECCV’24)

Họ cũng phê bình ADC và đề xuất **pixel‑error driven** densification + kiểm soát số primitive + sửa bias opacity khi clone. [arXiv+1](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)

**Điểm bạn có thể vượt**:

* họ dùng pixel error map làm criterion “auxiliary”; bạn dùng **$\delta I$ từ backprop** và **tích phân theo footprint** theo SRM Theorem 1 → story “theory‑consistent” hơn.
* nếu bạn chứng minh oracle-quality (phần dưới), bạn có thể claim: “not just pixel error, but _kernel‑correlated gradient_”.

### (C) Pixel‑GS (ECCV’24)

Pixel‑GS chỉ ra growth condition của 3DGS dùng **average gradient magnitude** nên “không grow đủ” cho large Gaussians nhìn từ nhiều view nhưng chỉ cover boundary pixels; họ sửa bằng **pixel‑aware weighting theo số pixel covered**. [ECVA+2Pixel-GS+2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf?utm_source=chatgpt.com)

**Điểm bạn có thể vượt**:

* score của bạn tự nhiên “pixel‑aware” vì footprint lớn → cover nhiều tiles → tích phân lớn (khi error tồn tại).
* bạn không chỉ weight theo pixel count; bạn weight theo **tương quan error signal với footprint**, sát “gradient theo v” hơn.

### (D) AbsGS (2024)

AbsGS phân tích “gradient collision” làm large Gaussians không split được; họ dùng homodirectional view‑space positional gradient. [arXiv+1](https://arxiv.org/abs/2404.10484?utm_source=chatgpt.com)

**Điểm bạn có thể vượt**:

* Nếu bạn thiết kế aggregator tránh cancellation (xem phần ablation), bạn có thể giảm collision _mà vẫn giữ link Theorem 1_ (ít nhất như robust surrogate).

### (E) Perceptual‑GS (2025)

Perceptual‑GS densify theo perceptual sensitivity để cân bằng chất lượng/efficiency. [arXiv+1](https://arxiv.org/abs/2506.12400?utm_source=chatgpt.com)

**Điểm bạn có thể “ăn”**:

* method bạn dễ generalize: chỉ cần đổi loss (ví dụ thêm LPIPS), $\delta I$ tự đổi theo backprop; kernel tích phân vẫn chạy. (Perceptual‑GS là baseline thú vị nếu bạn muốn claim “plug‑in loss”.)

---

## 2) Metrics: đo cái gì để “đúng mạch” với claim của bạn?

### 2.1 Metrics chuẩn NVS (ai cũng expect)

* **PSNR / SSIM / LPIPS** (để so với 3DGS, Pixel‑GS, AbsGS, Revising…). [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf?utm_source=chatgpt.com)
* (Tuỳ paper) thêm MS‑SSIM hoặc DISTS, nhưng PSNR/SSIM/LPIPS là đủ “đúng bài”.

### 2.2 Metrics efficiency & budget (đây là nơi densification ăn điểm)

* **#Gaussians cuối** (N)
* **VRAM peak**, **model size**
* **FPS render** (3DGS nhấn mạnh real‑time). [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)
* **Training wall‑clock** đến quality target (ví dụ PSNR = X hoặc LPIPS = Y)

### 2.3 Metrics “densification quality” (cái reviewer sẽ rất thích nếu bạn làm)

Bạn nên đo **quality per primitive** và **quality per time**:

* **PSNR vs #Gaussians** (curve)
* **LPIPS vs #Gaussians** (curve)
* **PSNR vs wall‑clock** (curve)
* **AUC(PSNR‑time)** hoặc “time‑to‑reach PSNR threshold”

Densification paper mà chỉ báo “final PSNR” thường bị hỏi: “có phải chỉ vì bạn tạo nhiều Gaussians hơn không?” → curve này đập thẳng câu hỏi đó.

### 2.4 Metrics “theory verification” (để chứng minh bạn không bịa)

Bạn cần metrics kiểm tra **score có thực sự là proxy $\|\nabla_{v_i}F\|$** không:

* **Corr(score_i, ||∂L/∂features_i||)** (Spearman/Kendall)
* **Corr(score_i, actual loss decrease nếu densify i)**
* **Top‑K hit rate**: K Gaussian bạn chọn có tạo ra “gain” thực sự không?

Những metric này _đúng tinh thần Frank–Wolfe_: chọn atom bằng LMO dựa trên gradient; và trong convex FW, **duality gap** là certificate đo chất lượng nghiệm. Jaggi 2013 nhấn mạnh duality gap là thước đo dễ tính và upper‑bound primal error (trong convex setting). [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v28/jaggi13.pdf?utm_source=chatgpt.com)
Bạn không có convex guarantee trong 3DGS, nhưng vẫn dùng được tinh thần “certificate proxy”.

---

## 3) Thiết kế thí nghiệm theo 3 tầng

### Tầng 1 — Sanity checks (để chắc chắn code bạn đang đo đúng “δI×ρ”)

**E1. Kiểm tra `image.grad` có đúng $$\delta I$$**

* Setup: chạy 1–2 iteration trên 1 scene nhỏ.
* Bắt buộc: `image.retain_grad()` trước backward.
* Log:
    * norm của `image.grad`
    * so sánh `image.grad` khác 0 khi loss khác 0
* Visualize: heatmap norm(grad) trên ảnh (đừng fancy, chỉ cần cho thấy nó tập trung ở vùng lỗi).

**E2. Kiểm tra footprint mapping đúng với rasterizer**

* Lấy vài Gaussian (random) → log:
    * projected pixel (mean2D) và rect_min/max tile bạn tính
    * số tiles touched
* Visualize overlay rect lên ảnh (debug mode).
* Nếu rect sai → score sai → tất cả “theory” bay màu.

**E3. Corr(score_i, grad w.r.t Gaussian appearance)**

* Với 3DGS, “amplitude/output vector” $v$ gần nhất là:
    * `features_dc` + `features_rest` (SH coefficients), và/hoặc color.
* Sau backward:
    * `g_true_i = ||∂L/∂features_i||`
    * `g_fw_i = score_i` của bạn (tile‑integrated)
* Report:
    * Spearman ρ_s giữa g_fw và g_true
    * Baseline: corr giữa criterion gốc (||∂L/∂mean2D||) và g_true

**Giải thích tại sao đúng-theory**: SRM Theorem 1 nói $\nabla_{v_i}F$ là tích phân của error signal $\delta F$ qua $\rho_i$. Ở NVS, $\delta F$ chính là $\partial\mathcal{L}/\partial I$ từ backprop (Example 1).
Nếu corr thấp → hoặc bug tile/projection, hoặc approximation tile quá thô, hoặc occlusion/transmittance làm sai proxy.

---

### Tầng 2 — “Oracle quality” experiments (đây là phần thuyết phục reviewer nhất)

Đây là nơi bạn chứng minh: **score của bạn chọn densify tốt hơn baseline**.

#### E4. “One‑step densification gain test” (chọn K ứng viên, xem ai giảm loss nhiều hơn)

Tại

Splat Regression Models

eckpoint k:

1. Freeze toàn bộ model hiện tại.
2. Tạo **candidate set** C (ví dụ 5k Gaussians visible).
3. Với mỗi method m (ADC, Revising, Pixel‑GS, AbsGS, Ours):
    * chọn top‑K gaussians theo criterion của m
4. Với mỗi gaussian i trong top‑K:
    * clone/split đúng như 3DGS (nhưng chỉ cho **1 gaussian** để test, hoặc mỗi lần test K nhỏ)
    * chạy **t** steps SGD ngắn (ví dụ 10–20 iters) chỉ optimize:
        * new gaussian params (appearance + covariance + position)
        * hoặc optimize all but với t rất nhỏ
    * đo:
        * ΔLoss = Loss_before − Loss_after_t
5. Report:
    * mean ΔLoss của top‑K
    * median ΔLoss (robust)
    * variance

**Điểm mấu chốt**: test này cô lập “chất lượng chọn densify”, không bị confound bởi training dài.

**Vì sao nó gắn với Frank–Wolfe?**
Trong FW, LMO chọn atom theo gradient để giảm objective nhiều nhất theo xấp xỉ bậc nhất. Nếu score của bạn ~ $\|\nabla_{v_i}F\|$, ranking theo score phải cho **gain** tốt hơn. (Dù nonconvex, ta vẫn test được “local first‑order usefulness”.) [Proceedings of Machine Learning Research+1](https://proceedings.mlr.press/v28/jaggi13.html?utm_source=chatgpt.com)

#### E5. “Score‑gain correlation” (đúng nghĩa verify theory)

Trên cùng checkpoint:

* Lấy một batch lớn candidate i.
* Tính:
    * `score_i` (yours)
    * `gain_i` (ΔLoss sau t steps như E4)
* Report:
    * Spearman corr(score, gain)
    * Top‑p precision: trong top 1% score, bao nhiêu % nằm trong top 1% gain?

Nếu corr tốt và stable across scenes → reviewer sẽ tin bạn “not just heuristic”.

#### E6. Cancellation / occlusion diagnostic (để không bị “AbsGS-style” bắt bẻ)

AbsGS nói gradient collision làm large gaussians không split. [arXiv](https://arxiv.org/abs/2404.10484?utm_source=chatgpt.com)
Bạn cần chứng minh bạn không “tái phát” theo cách khác:

* Decompose score theo view (mỗi training iteration là 1 view/cam):
    * $g_i^{(v)}$ = integrated δI×ρ trên view v
* Define cancellation ratio: $$r_i = \frac{\|\sum_v g_i^{(v)}\|}{\sum_v \|g_i^{(v)}\|}$$
* Thí nghiệm:
    * thống kê r_i cho những Gaussian bị “blurry region” vs “sharp region”
    * compare baseline ADC vs Ours

Nếu r_i rất nhỏ ở vùng cần split → bạn nên dùng aggregator “sum of norms” (xem ablation) hoặc gating.

---

### Tầng 3 — Benchmarks đầy đủ (để “paper-ready”)

#### E7. Standard datasets & metrics

Dùng bộ benchmark mà cộng đồng 3DGS hay dùng:

* **Mip‑NeRF 360**, **Tanks & Temples**, **Deep Blending**
    (Pixel‑GS và PUP 3D‑GS đều nhắc tới các dataset này; 3DGS gốc cũng report trên established datasets). [arXiv+2HKU Scholars Hub+2](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

Report:

* PSNR/SSIM/LPIPS
* #Gaussians, VRAM, FPS (render)
* training time tổng

#### E8. Budget‑matched evaluation (để tránh câu hỏi “bạn chỉ tăng N thôi?”)

Chạy 2 kiểu:

1. **Fixed time / fixed iterations**: ai đạt PSNR/LPIPS tốt hơn?
2. **Fixed #Gaussians budget**: ép mỗi method kết thúc ở N gần như nhau:
    * Revising Densification có cơ chế kiểm soát tổng primitives [arXiv+1](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)
    * bạn cũng có thể densify theo top‑K với K được set để giữ budget

Sau đó vẽ:

* PSNR vs #Gaussians
* LPIPS vs #Gaussians
    Chỗ này nếu bạn thắng, bạn thắng “đúng bản chất densification”.

#### E9. Convergence curves

Plot theo wall‑clock:

* PSNR(t), LPIPS(t)
* #Gaussians(t)

Kỳ vọng của bạn: **ít densify “lãng phí” hơn → đạt quality target sớm hơn**.

---

## 4) Ablation bắt buộc (để reviewer không nói “thắng do tweak linh tinh”)

Bạn nên ablate đúng các giả định “theory → implementation”:

### A1. Dùng `image.grad` vs dùng |I−GT|

* Version (theory‑clean): δI = image.grad (đúng Example 1)
* Version (heuristic): δI ≈ I−GT
    So sánh E4/E5: nếu `image.grad` cho oracle-quality tốt hơn, bạn có bằng chứng “theory matters”.

### A2. Tile weight $w_{i,t}$

* w=1 (bbox uniform)
* isotropic Gaussian weight theo distance tile center (σ=r/3)
* (optional) approximate anisotropy nếu bạn có conic

Đo:

* corr(score, gain)
* overhead runtime

### A3. Aggregation để tránh cancellation

* `score = ||Σ_t w G_t||` (norm of sum)
* `score = Σ_t ||w G_t||` (sum of norms)
* per-view sum of norms: `Σ_v ||g_i^(v)||`

AbsGS đã chỉ ra cancellation là

Splat Regression Models

[arXiv](https://arxiv.org/abs/2404.10484?utm_source=chatgpt.com)
Bạn có thể trình bày như “robust surrogate”: vẫn dựa trên cùng integrand δI×ρ, chỉ đổi norm/aggregation để tăng stability.

### A4. Gating occlusion (để tránh chọn Gaussian “bị che”)

* baseline: visibility_filter
* * gating bằng grad mean2D norm (như bạn đã dự kiến)
* * gating bằng estimated transmittance (nếu bạn đọc được từ renderer)

Đo lại E4/E5: gating nào giúp top‑K gain cao nhất.

---

## 5) Stress tests “đánh đúng các paper gần đây”

### S1. Poor initialization test (đánh vào Pixel‑GS)

Pixel‑GS nhấn mạnh 3DGS phụ thuộc mạnh vào chất lượng point cloud init; poor init gây blur/needle artifacts; họ sửa bằng pixel‑aware gradient weighting. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf?utm_source=chatgpt.com)

Thiết kế:

* subsample COLMAP points còn 50%, 25%, 10%
* hoặc add noise vào points
    So sánh:
* tốc độ phục hồi chi tiết (LPIPS giảm nhanh hơn?)
* số Gaussians cần để đạt PSNR target

### S2. High‑frequency detail scenes (đánh vào AbsGS)

AbsGS target vùng “intricate, high‑frequency” và nói gradient collision gây blur. [arXiv](https://arxiv.org/abs/2404.10484?utm_source=chatgpt.com)
Thiết kế:

* chọn scenes nhiều chi tiết (fences, leaves, text)
* thêm metric phụ: PSNR/LPIPS chỉ trên edge mask (Sobel trên GT) để đo “detail recovery”

### S3. Primitive budget control (đánh vào Revising Densification)

Revising Densification nói họ kiểm soát tổng primitives + sửa bias opacity clone, và cải thiện chất lượng ổn định. [arXiv+1](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com)
Thiết kế:

* ép budget N giống nhau
* so sánh quality + distribution gaussian sizes (radii histogram)

---

## 6) “Implementation-aware” profiling (để reviewer không hỏi “thêm kernel có làm chậm không?”)

Bạn cần report overhead rất thẳng:

* time per iter (forward+backward)
* time extra cho:
    * tile_reduce kernel
    * fw_score kernel
* chạy score kernel **chỉ ở densify interval** (giống 3DGS densify period), không phải mọi iter
* report “% overhead” trên tổng training time

3DGS gốc nhấn mạnh real-time rendering/training efficiency; nên efficiency claim phải đo rõ. [arXiv](https://arxiv.org/abs/2308.04079?utm_source=chatgpt.com)

---

## 7) Kỳ vọng kết quả (và nếu không ra, bạn biết “vì sao”)

### Nếu method bạn thực sự tốt, thường bạn sẽ thấy:

* **E4/E5**: top‑K của bạn cho ΔLoss lớn hơn baseline (đây là “smoking gun”).
* corr(score, gain) cao hơn ADC.
* Convergence curve: đạt PSNR/LPIPS target sớm hơn hoặc với ít Gaussians hơn.

### Nếu không tốt, 3 nguyên nhân hay gặp (và thí nghiệm nào bắt được):

1. **Bug footprint mapping / tile rect** → E2/E3 corr thấp bất thường.
2. **Occlusion/transmittance mismatch** → E4 top‑K gain kém trong scene occluded; A4 gating sẽ cứu.
3. **Cancellation do aggregation** → E6 cancellation ratio thấp; A3 (sum of norms) sẽ cải thiện.

---

## 8) Bộ “core figures” nên có trong paper (rất thuyết phục reviewer)

1. Scatter plot: **score vs gain** (E5) + Spearman corr.
2. Bar plot: mean/median ΔLoss của top‑K (E4) so với baselines.
3. Curves: **LPIPS vs #Gaussians** (budget‑quality).
4. Curves: **PSNR vs time** (convergence).
5. Runtime overhead table (ms/iter, %).
6. Qualitative: error map + overlay densified regions.

---

Tóm lại: nếu bạn làm đúng **E4/E5** (oracle-quality) và thắng, bạn gần như đã có “reviewer-proof core contribution”, vì bạn đang chứng minh đúng thứ mà SRM Theorem 1 gợi ý: _chọn splat theo correlation của error signal với kernel footprint_ , thay vì heuristic. Và nó còn đối thoại trực tiếp với xu hướng cộng đồng: pixel‑error densification (Revising) [arXiv+1](https://arxiv.org/abs/2404.06109?utm_source=chatgpt.com), pixel coverage (Pixel‑GS) [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf?utm_source=chatgpt.com), chống collision (AbsGS) [arXiv](https://arxiv.org/abs/2404.10484?utm_source=chatgpt.com).

Nếu bạn muốn tối ưu đường đi đến publish A*: ưu tiên implement logging để làm **E4/E5** trước (nó là “unit test” cho ý tưởng), rồi mới chạy benchmark full.

Splat Regression Models

* * *

**You:**

Tôi đã implement nhưng có những vấn đề tôi cần bạn phân tích và suy nghĩ sâu hơn, vì đề xuất implement hiện tại đang không thể vượt qua baseline, hãy phân tích và tìm hiểu xem hướng để phát triển bài toán theory implementation tốt hơn. Hãy phân tích report của tôi dựa trên những gì tôi đã làm và những vấn đê gặp phải sau đó hãy suy nghĩ hướng tốt và refine lại pipeline cho chính xác. Tập trung nhất vào phần 6.0, và kiểm tra verify các phần còn lại nhé. “””
**1) Mục tiêu theory**
- Thay densification heuristic của 3DGS bằng tiêu chí **FW / tile‑integrated image‑gradient** dựa trên SRM Theorem 1:
  - dùng **δI = ∂L/∂I** (image.grad sau backward),
  - tích phân (xấp xỉ) δI trên footprint của Gaussian,
  - lấy norm làm score để chọn densify.

**2) Pipeline đã triển khai**
- **Render → Backward → FW‑score**:
  1. Render ảnh.
  2. image.retain_grad() trước backward.
  3. Sau backward, lấy image.grad làm δI.
  4. CUDA kernel **tile‑reduce** δI thành tile sums.
  5. CUDA kernel **compute FW score** theo tile rect/bbox đúng rasterizer.
  6. Accumulate score vào buffer fw_score_accum/denom.
  7. Densify bằng score (thay heuristic grad mean2D).

- **Rasterizer consistency**:
  - FW score tính đúng theo **tile rectangle & projection** giống rasterizer (ndc2Pix/getRect/bbox).
  - Score được tính trong no_grad để không ảnh hưởng graph.

**3) Thay đổi chính (theory‑relevant)**
- **CUDA**: thêm kernel tile‑reduce + fw‑score (không dùng avg_pool2d).
- **Renderer**: trả geomBuffer, binningBuffer để fw‑score dùng trực tiếp.
- **Training loop**: tính fw_score sau backward; dùng grads_override trong densify.
- **GaussianModel**: thêm fw_score_accum, fw_denom.

**4) Heuristic bắt buộc vẫn còn**
- Selection vẫn cần threshold/top‑K (không thể tránh): đây là **heuristic** để chọn densify set.
- Hiện dùng percentile threshold (có schedule) để kiểm soát tốc độ densify.

**5) Trạng thái đúng‑theory**
- **Đúng phần “score construction”**: δI × footprint → tích phân → norm.
- **Xấp xỉ**: tile‑sum bỏ qua transmittance/occlusion (chấp nhận như proxy).
- **Selection vẫn heuristic**: đây là điểm cần chấp nhận và nêu rõ trong report.

6.0 Overhead cao khi point cloud lớn
* Triệu chứng: tốc độ huấn luyện giảm mạnh khi số Gaussian tăng; chậm hơn baseline 3DGS rõ rệt.
* Giả thiết chính:
    * N tăng nhanh → rasterizer cost tăng (O(N) + binning), đây là nguyên nhân chính.
    * FW pipeline thêm 2 kernel + buffers → overhead phụ.
    * Densify “burst” làm N tăng nhanh hơn baseline.
* Cần xác nhận: overhead thuần của FW kernel khi N cố định; tốc độ giảm có tương quan trực tiếp với N.
6.1 “Không heuristic” vs thực tế cần heuristic selection
* Lý thuyết: score = ∫ δI ρ (FW‑style, SRM Theorem 1).
* Thực tế: cần threshold/top‑K để chọn densify → heuristic selection.
* Cần chốt: chấp nhận heuristic selection như restricted LMO, hay định nghĩa LMO top‑K cố định để giảm heuristic?
6.2 FW score có đúng proxy ∇v không?
* Lý thuyết: ∇v ~ ∫ δI ρ.
* Thực tế: tile‑sum + bỏ qua transmittance/occlusion → proxy có bias.
* Cần xác nhận: mức bias do occlusion, có làm sai ranking không?
6.3 Chọn threshold thế nào là đúng?
* Thực tế: percentile (P90/P95/P99) vẫn gây “burst” densify ở vài iter.
* Nguyên nhân khả dĩ: score distribution zero‑inflated hoặc thay đổi mạnh theo iter.
* Cần xác nhận: phân phối score theo iter, tỷ lệ valid_norm.
6.4 Ít point cloud nhưng chất lượng vẫn không vượt baseline
* Triệu chứng: ngay cả khi số points được giữ thấp, PSNR/SSIM/LPIPS vẫn kém baseline 3DGS.
* Ý nghĩa: vấn đề không chỉ do “densify quá nhiều”, mà còn có thể do tiêu chí chọn densify chưa đúng hoặc proxy không đủ tương quan với ∇v thật.
* Cần xác nhận:
    * Corr(score, ‖∂L/∂features‖) có đủ cao không?
    * Top‑K score có thực sự tạo gain tốt hơn baseline? “””

* * *

**ChatGPT:**

Mình đọc kỹ report của bạn và… “mùi” khoa học ở đây là thật: bạn đã đi đúng đường **ở mức skeleton** (lấy `δI = ∂L/∂I` sau backward, tile‑reduce, rồi tích phân trên footprint để ra score, tính trong `no_grad` để không phá graph). Nhưng việc **không vượt baseline** (thậm chí khi giữ N thấp) cộng với **overhead tăng mạnh** gần như chắc chắn đến từ vài “mismatch” rất cụ thể giữa _theory mà bạn muốn mô phỏng_ và _thứ CUDA kernel của bạn đang tính_. Mình sẽ bóc tách theo đúng các mục 6.0–6.4 của bạn, và ở mỗi chỗ mình sẽ chỉ ra: cái nào là **theorem‑consistent**, cái nào là **proxy hợp lý**, cái nào là **rủi ro khiến ranking sai / không giúp densify**.

---

## 0) Sanity check: bạn đang “implement đúng ý tưởng” nhưng chưa chắc “đúng theorem”

Trong SRM, Theorem 1 cho Wasserstein gradient theo tham số (v, A, b) của splat measure. Điểm quan trọng nhất với hướng bạn đang làm là:

* **Wasserstein gradient theo v** có dạng (rút gọn):

$$\nabla_v F(f_\mu)(v,A,b) \;=\; \int \delta F(x)\,\rho_{A,b}(x)\,\pi(dx)$$

tức **lấy “error signal” $\delta F$** rồi **lọc (kernel average) bằng density $\rho_{A,b}$**.

Splat Regression Models

* Trong Example 1 của SRM, $\delta F(x_i)$ chính là “error signal” kiểu backprop ở sample $x_i$.

    Splat Regression Models


Bạn map sang 3DGS như sau:

* coi **pixel** là “x”,
* lấy $\delta I=\partial L/\partial I$ (image.grad) làm $\delta F$,
* footprint của Gaussian trên screen ~ kernel $\rho$.

Về _tinh thần_ thì ổn.

Nhưng có 2 điều SRM “ép” bạn phải tôn trọng nếu muốn nói “đúng Theorem 1” chứ không chỉ là “heuristic nghe hợp lý”:

1. **$\rho$ là density chuẩn hoá** (tích phân = 1). Nếu bạn cộng gradient trên bbox/tile mà **không weight bằng $\rho$** hoặc **không normalize**, bạn đang tính một thứ tỉ lệ theo diện tích footprint → lệch hẳn khỏi $\int \delta F \rho$.
2. $\delta F$ trong NVS “đúng” còn có chuyện **adjoint của renderer** (chain rule qua alpha‑compositing / visibility). Bạn đã chấp nhận proxy bỏ qua transmittance/occlusion, ok, nhưng phải ý thức: ranking có thể sai rất nặng trong cảnh nhiều che khuất.

Hai mismatch này giải thích thẳng vào 6.2 và 6.4.

---

## 6.0 Overhead cao khi point cloud lớn: nguyên nhân “chắc chắn đúng” + nơi bạn có thể thắng

### 6.0.a Cái “đúng chắc” (không cần đo cũng biết)

3DGS rasterizer tile‑based, mỗi frame cost tăng theo:

* số Gaussian N,
* và quan trọng hơn: **#intersections (Gaussian × tiles/pixels)**.

Trong paper 3DGS họ nhấn mạnh renderer “visibility‑aware” và tile list traversal front‑to‑back để nhanh realtime. [arXiv+1](https://arxiv.org/abs/2308.04079)
Khi N tăng lên hàng triệu, chi phí binning + traversal list sẽ dominate.

Bạn đang gặp đúng triệu chứng kinh điển: N tăng → rasterizer chậm → training chậm. Không bất ngờ.

### 6.0.b Nhưng tại sao **chậm hơn baseline rõ rệt**?

Ở đây có 3 nghi phạm “rất đời”:

**(i) Bạn đang tính FW score mỗi iteration, trong khi densification chỉ cần mỗi ~100 iters**
Baseline ADC thu thập stats rồi densify theo schedule (Revising Densification tóm lại rất rõ ADC chạy theo lịch, dựa trên gradient stats tích luỹ). [ar5iv](https://ar5iv.org/pdf/2404.06109)
Nếu bạn chạy tile‑reduce + FW kernel _mỗi iter_ thì bạn tự cộng thêm một cost O(HW) + O(N·#tiles_rect) cho toàn training. Với resolution kiểu 800×800 trở lên, O(HW) là “đau” thực sự.

**Fix (rất nên):** chỉ compute tile‑reduce + FW score đúng vào window thu thập stats (ví dụ mỗi 100 iters), hoặc compute thưa (1/4 iters) rồi EMA.

---

**(ii) Kernel FW của bạn có thể đang O(N·area(rect_tiles))**
Nếu mỗi Gaussian bạn duyệt qua tất cả tiles trong bbox → với Gaussian lớn (radii lớn) cost tăng theo diện tích rect. Khi scene nhiều Gaussian lớn, kernel thành “đốt GPU”.

**Fix “giết overhead” mà vẫn giữ idea: prefix‑sum trên grid tiles**
Bạn đã có tile_sums (Tx×Ty×3). Nếu score của bạn là **box‑integral** trên rect, thì bạn có thể:

* tạo integral image / prefix sum trên tile grid cho mỗi channel,
* mỗi Gaussian lấy rect sum bằng **4 lần đọc** (inclusion‑exclusion),
* complexity: O(Tx·Ty) để prefix + O(N) để query.

Cực rẻ vì Tx,Ty nhỏ (W/16,H/16).
Điều này _giữ nguyên_ proxy của bạn, chỉ thay cách tính để giảm wall‑clock.

(Chưa weight $\rho$ được, nhưng giải quyết 6.0 rất mạnh.)

---

**(iii) “Trả geomBuffer/binningBuffer ra Python” có thể tạo overhead do allocation/lifetime**
Revising Densification cũng chỉ ra ADC dựa trên stats tích luỹ; không cần expose toàn bộ internal buffers ra Python mỗi iter. [ar5iv](https://ar5iv.org/pdf/2404.06109)
Nếu bạn return các buffer lớn, PyTorch có thể:

* giữ reference,
* gây nhiều allocation,
* làm tăng pressure memory → giảm throughput.

**Fix:** tốt nhất là **tính FW score ngay trong rasterizer (C++/CUDA)** (fused hoặc side‑output nhỏ), hoặc chỉ return **rect info** (pix_min/pix_max) + depth/opacity scalar cần thiết, thay vì entire binning list.

---

### 6.0.c Densify “burst” làm N tăng nhanh hơn baseline

Cái này bạn đã tự chẩn đoán đúng. Percentile threshold dễ tạo burst vì distribution score “zero‑inflated + heavy‑tail” (6.3). Khi burst, N nhảy, rasterizer cost nhảy.

**Fix đồng bộ với theory FW:** dùng **top‑K cố định** (budget) mỗi densify run. Đây không chỉ là “heuristic” mà còn _rất hợp với Conditional Gradient/Frank–Wolfe_: LMO chọn (một vài) atom tốt nhất mỗi bước.
Và cộng đồng densification cũng đang đi theo hướng

Splat Regression Models

ve” (Revising Densification có hẳn cơ chế limit tổng primitive và limit số primitive mỗi lần densify). [ar5iv](https://ar5iv.org/pdf/2404.06109)

---

## 6.1 “Không heuristic” vs thực tế cần heuristic selection

Bạn đang đúng khi nói “threshold/top‑K là heuristic selection”.

Nhưng để bài của bạn “đỡ bị reviewer bắt bẻ”, bạn nên **đóng khung đúng kiểu optimization**:

* Bạn không claim “no heuristic”.
* Bạn claim: **Restricted LMO**.

Tức là:

* không tối ưu trên không gian liên tục mọi (A,b),
* mà LMO chỉ search trong tập ứng viên “gắn” vào splats hiện có (split/clone candidates).

Đây là cách Revising Densification cũng “chấp nhận thực tế”: họ đưa per-pixel error về per‑Gaussian score và vẫn phải chọn subset để grow, thậm chí còn có budget và ranking theo score. [ar5iv](https://ar5iv.org/pdf/2404.06109)

**Điểm mấu chốt:**

* Percentile threshold là heuristic “khó ổn định”.
* Top‑K fixed là heuristic “giống LMO nhất” + dễ kiểm soát compute.

---

## 6.2 FW score có đúng proxy ∇v không? (và vì sao ranking có thể sai nặng)

Bạn viết: “tile‑sum bỏ qua transmittance/occlusion → proxy có bias”. Mình đồng ý, và mức bias này **có thể đủ để giết performance**.

### 6.2.a Mismatch 1: Bạn có đang weight bằng $\rho$ thật không?

Trong Theorem 1, $\nabla_v$ là **kernel average**: $\int \delta F(x)\rho(x)$.
Nếu bạn làm:

* tile_reduce: $G_t=\sum_{u\in tile} \delta I(u)$
* gaussian_score: $S_i=\sum_{t\in rect(i)} G:contentReference[oaicite:10]{index=10}\|$

thì bạn vừa:

* dùng **box kernel** thay vì Gaussian kernel,
* và quan trọng hơn: bạn đang tính **integral không chuẩn hoá**, nên Gaussian có rect lớn sẽ được ưu tiên chỉ vì “ăn nhiều pixel”.

Đây là lỗi “theorem‑breaking” số 1.

**Theorem‑consistent fix (rất quan trọng):**
Tối thiểu phải normalize:

$$\hat g_i = \frac{\sum_{t\in rect(i)} w_{i,t} G_t}{\sum_{t\in rect(i)} w_{i,t} + \varepsilon}$$

* nếu chọn $w_{i,t}=1$ thì $\hat g_i$ là **mean δI trong rect**, không còn bias diện tích.
* tốt hơn: $w_{i,t}\approx\int_{tile} \rho_i(u)\,du$ (evaluate Gaussian conic tại tile center × tile area).

Nếu bạn **chưa** làm normalization/weight, thì việc không vượt baseline là… hoàn toàn hợp lý, vì ranking đang “ưu tiên Gaussian to” theo cách không đúng với SRM.

### 6.2.b Mismatch 2: Occlusion / transmittance làm “δI ở đây” không thuộc về Gaussian đó

Revising Densification khi map pixel error về per‑Gaussian error họ _redistribute theo contribution_ $w_{k,i}(p)$ (tức weight theo đóng góp thật của Gaussian vào pixel). [ar5iv](https://ar5iv.org/pdf/2404.06109)
Đó chính là cách họ tránh “đổ lỗi pixel cho sai Gaussian”.

Bạn hiện đang làm “đổ lỗi” theo footprint geometry (bbox), không theo contribution. Khi cảnh nhiều layer alpha/occlusion, bạn sẽ pick sai primitive.

**Occlusion gating rẻ mà implement‑được:**

* Render thêm depth map (nhiều fork rasterizer hỗ trợ; Revising Densification cũng nói về alpha‑compositing/residual transmittance). [ar5iv](https://ar5iv.org/pdf/2404.06109)
* Với mỗi Gaussian i, lấy depth $z_i$ ở camera space, sample depth map tại mean2D (hoặc tile center gần mean2D):
    * nếu $z_i > depth(p_i)+\tau$ → likely occluded → downweight score (0 hoặc sigmoid).
        Cái này không cần per‑pixel transmittance list.

**Lưu ý:** đây là proxy, nhưng ít nhất nó làm ranking đỡ “ảo” khi Gaussian nằm sau surface.

### 6.2.c Mismatch 3: δI của loss (SSIM/L1) có thể noisy + sign‑cancellation

δI có dấu (+/-) theo kênh và pixel. Khi bạn sum, cancellation xảy ra. Điều này liên quan trực tiếp đến 6.4.

---

## 6.3 Threshold/percentile gây burst: đúng, và có cách “theory-consistent” hơn

Revising Densification liệt kê limitation của ADC: threshold positional gradient **nhạy** và không kiểm soát được #primitives (có thể OOM). [ar5iv](https://ar5iv.org/pdf/2404.06109)
Pixel‑GS cũng chỉ ra “merely lowering threshold” gây tăng điểm không cần thiết mà vẫn blur ở vùng thiếu points. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

Percentile schedule của bạn là một dạng “lowering threshold” dynamic → dễ burst.

**Giải pháp tốt nhất (vừa hợp theory FW vừa hợp cộng đồng):**

* chọn **top‑K fixed per densify run**,
* có **global cap N_max**,
* và nếu vượt cap thì “replace pruned by spawned” hoặc prune theo score thấp.

Đây chính là cơ chế Revising Densification dùng: global limit + budget mỗi densify run, và nếu số ứng viên vượt budget thì giữ score cao nhất. [ar5iv](https://ar5iv.org/pdf/2404.06109)

Bạn có thể nói thẳng trong paper: “our LMO returns top‑K atoms under a compute/memory budget”.

---

## 6.4 Ít point cloud nhưng chất lượng vẫn kém baseline: đây mới là “điểm chết” và nguyên nhân sâu

Nếu giữ N thấp mà vẫn kém baseline, thì **không phải chỉ do burst/overhead**. Nó nói rằng:

> “score bạn tối ưu hoá đang chọn sai chỗ để densify / hoặc densify đúng chỗ nhưng không giúp giảm loss”.

Mình thấy có một nguyên nhân “rất có theory” và khớp với SRM + thực tế 3DGS:

### 6.4.a Bạn đang dùng proxy của **∇v**, nhưng densification (đặc biệt split) cần phát hiện **high-frequency residual trong footprint**, không phải mean residual

Trong SRM, ∇v là kernel‑average của δF.
Giờ tưởng tượng một Gaussian lớn che lên vùng biên edge: nửa footprint error dương, nửa âm. Khi lấy average, nó gần 0 → ∥∇v∥ nhỏ → bạn **không densify**.
Nhưng chính trường hợp này _lại là_ trường hợp cần split để bắt high‑frequency detail.

Revising Densification cũng nêu một limitation rất

Splat Regression Models

adient “blind to absolute error” và có kịch bản high‑frequency pattern phủ bởi vài Gaussian lớn, gradient vị trí gần 0 nên không trigger densify dù error lớn. [ar5iv](https://ar5iv.org/pdf/2404.06109)
AbsGS gọi một hiện tượng họ phân tích là “gradient collision” làm Gaussian lớn không split và dẫn tới blur. [arXiv+1](https://arxiv.org/abs/2404.10484)
Pixel‑GS cũng nói averaging gradient across views làm large Gaussian khó grow. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

=> Kết luận: **chỉ dùng ∥∇v∥ (mean) để quyết định densify rất dễ fail ở vùng chi tiết**.

### 6.4.b “Fix có lý thuyết + implement được” nhất: thêm **energy/variance score** (second-moment) để detect cancellation

Bạn đã có tile_reduce. Bạn có thể sửa rất nhẹ:

* Thay vì chỉ giảm $\delta I$ thành vector sum per tile:
    * $S_t = \sum_{u\in tile} \delta I(u)$ (vector 3)
* Thêm một scalar:
    * $Q_t = \sum_{u\in tile} \|\delta I(u)\|^2$ (scalar)

Sau đó cho Gaussian i (có weights w_{i,t}, và normalize):

$$m_i = \frac{\sum_t w_{i,t} S_t}{\sum_t w_{i,t}+\varepsilon} \quad\text{(mean vector)}$$ $$e_i = \frac{\sum_t w_{i,t} Q_t}{\sum_t w_{i,t}+\varepsilon} \quad\text{(mean energy)}$$ $$\mathrm{var}_i = e_i - \|m_i\|^2 \quad\text{(proxy “variance” / cancellation)}$$

**Interpretation:**

* $\|m_i\|$ ~ magnitude of ∇v (Theorem 1 style).
* $\mathrm{var}_i$ lớn nghĩa là δI biến thiên mạnh trong footprint, mean bị triệt tiêu → representation thiếu tần số cao → **split** nên được ưu tiên.

Cái này cực hợp “kể chuyện” với reviewer:

* bạn dùng Theorem 1 cho **first-order** signal (mean),
* và dùng **second-moment** để phát hiện _khi first-order đánh lừa do cancellation_ → densify cho chi tiết.

**Quan trọng:** đây không phải bịa đặt vô căn cứ; nó là một surrogate rất tự nhiên (dạng bias‑variance) và nó giải thích đúng failure mode blur/edge.

### 6.4.c Bạn cũng cần “semantic split vs clone”

Trong ADC, split/clone phụ thuộc size covariance (Revising Densification mô tả rõ). [ar5iv](https://ar5iv.org/pdf/2404.06109)
Bạn nên map score như sau:

* **Clone score** (cho Gaussian nhỏ): dùng $\|m_i\|$ (mean error signal)
* **Split score** (cho Gaussian lớn): dùng $\mathrm{var}_i$ (high‑freq residual) × maybe scale factor theo radius

Và chọn top‑K riêng cho clone và split.

Điều này còn ăn khớp với Pixel‑GS và AbsGS: họ đều đang cố làm large Gaussian dễ split/grow hơn trong vùng thiếu detail. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

---

## 7) Refinement proposal: “FW‑Moment Densification” (giữ idea của bạn, sửa đúng chỗ đang làm bạn thua baseline)

Mình viết rõ như một “spec” để bạn sửa code mà không lạc.

### Inputs bạn đã có (và nên giữ)

* `image.grad` sau backward: δI
* tile partition giống rasterizer: BLOCK_X/BLOCK_Y
* per Gaussian: rect tiles (pix_min/pix_max hoặc tile_min/tile_max), radius/scale, depth z, opacity

### Kernel 1: tile_reduce (chỉ chạy khi cần densify stats)

Output per tile:

* `S_t[Ty,Tx,3] = Σ δI(u)` (FP32 accum)
* `Q_t[Ty,Tx] = Σ ||δI(u)||^2` (FP32)

### Kernel 2: gaussian_score

Với mỗi Gaussian i:

* loop tiles trong rect (hoặc dùng prefix‑sum trick nếu w=1).
* compute weights `w_{i,t}`:
    * **option A (nhanh, box):** `w=1`
    * **option B (theory‑closer):** `w = exp(-0.5 * quadform(tile_center - mu2D, conic)) * tile_area`
* accumulate numerator vector + energy:
    * `num_v += w * S_t`
    * `num_e += w * Q_t`
    * `denom += w`

Then:

* `m = num_v / (denom + eps)`
* `e = num_e / (denom + eps)`
* `var = max(e - dot(m,m), 0)`

Occlusion gating (cheap):

* sample depth map at mu2D (nearest)
* if `z_i > depth_sample + τ`: `gate=0` else gate=1 (hoặc sigmoid)

Scores:

* `clone_score = gate * ||m||`
* `split_score = gate * var`

Accumulate theo iter:

* `clone_accum[i] += clone_score`, `clone_denom[i] += 1`
* `split_accum[i] += split_score`, `split_denom[i] += 1`

### Selection (mỗi densify run)

* `clone_rank = clone_accum/clone_denom` trên set “small gaussians”
* `split_rank = split_accum/split_denom` trên set “large gaussians”

Pick:

* top‑K_clone, top‑K_split (fixed budget)
* global cap N_max như Revising Densification [ar5iv](https://ar5iv.org/pdf/2404.06109)

### Densify action

* clone/split theo logic gốc 3DGS (giữ sampling covariance, v, v.v.)
* **Nhớ opacity bias khi clone**: Revising Densification chứng minh clone giữ opacity tạo bias do alpha‑compositing; họ đề xuất chỉnh opacity hậu clone để giảm bias. [ar5iv](https://ar5iv.org/pdf/2404.06109)
    → nếu bạn muốn “thắng baseline”, đây là một patch cực đáng: vì bạn đang thay densification criterion, nhưng nếu clone gây bias mạnh thì training dễ lệch.

---

## 8) Checklist verify nhanh để bạn biết mình đang sửa đúng bệnh (không tự thôi miên)

### (A) Verify “theorem‑consistency” (điểm bạn đang yếu nhất)

* [ ]  Bạn có **normalize** kernel integral không? (denom)
    Nếu không, bạn đang không mô phỏng $\int \delta F \rho$.
* [ ]  `w_{i,t}` có phụ thuộc conic/scale không? Nếu toàn 1, bạn đang dùng box kernel; acceptable như proxy nhưng phải **nói rõ**.
* [ ]  Bạn có log để chứng minh “large footprint bias” đã hết không? (corr(score, area(rect)) trước/sau)

### (B) Verify “ranking đúng Gaussian có đóng góp”

* [ ]  score của Gaussian occluded có giảm không?
    (log score vs depth residual)
* [ ]  fraction top‑K nằm trong vùng error map cao (overlay render error)

### (C) Verify “6.4 cancellation”

* [ ]  Tính thống kê:
    * mean norm: $\|m_i\|$
    * variance pro

        Splat Regression Models

        ne blur edge thường sẽ có **var_i cao ở Gaussian lớn**.
* [ ]  So sánh: split theo var_i vs split theo mean → PSNR/LPIPS ở vùng detail.

### (D) Verify “6.0 overhead”

* [ ]  kernel time: tile_reduce + gaussian_score chỉ chạy mỗi 100 iters
* [ ]  gaussian_score complexity có bị O(N·area) không? nếu có, cân nhắc prefix‑sum (w=1) hoặc candidate filtering.
* [ ]  N growth: top‑K fixed + global cap giống Revising Densification [ar5iv](https://ar5iv.org/pdf/2404.06109)

---

## Kết luận thẳng: bạn đang thua baseline vì 2 lỗi “có hệ thống”

1. **Score hiện tại rất dễ không đúng Theorem 1** nếu thiếu weight/normalize → ranking bị bias diện tích.
2. Dù có sửa (1), **chỉ dùng proxy của ∇v (mean)** là chưa đủ cho densification (đặc biệt split) vì cancellation/high‑freq residual. Điều này khớp với limitation mà Revising Densification phân tích (pos‑grad blind to absolute error) [ar5iv](https://ar5iv.org/pdf/2404.06109), và với các paper như AbsGS/Pixel‑GS đang cố chữa “large Gaussian không split/grow đúng chỗ”. [arXiv+1](https://arxiv.org/abs/2404.10484)

Nếu bạn implement “FW‑Moment” (mean + variance) + top‑K budget + (tùy) opacity correction khi clone theo Revising Densification, mình đánh giá xác suất vượt baseline sẽ **cao hơn hẳn** so với gi

Splat Regression Models

n tại mà chỉ tuning threshold. [ar5iv](https://ar5iv.org/pdf/2404.06109)

Phần hay là: sửa này không phá idea SRM của bạn; nó làm idea **đúng hơn với math** và **đúng hơn với failure mode cộng đồng đang chữa**.

* * *

**You:**

Uploaded a file
Hiện tại kết quả rất tệ, tôi sẽ báo cáo tình trạng hiện tại (FW densification – mean/variance) “”” Kết quả hiện tại
* Ours: SSIM 0.5403, PSNR 20.91, LPIPS 0.4502
* Baseline 3DGS (tham chiếu trước): SSIM ~0.766, PSNR ~25.22, LPIPS ~0.209 → FW densification vẫn thấp hơn baseline rõ rệt, dù đã thêm mean/variance và top‑K.
Mình đã implement đúng skeleton theo SRM/Theory
* δI = ∂L/∂I sau backward.
* Tile reduction → integrate over footprint.
* Kernel weight theo conic + normalize theo sum_w.
* Tách clone (mean score) và split (variance score).
* Top‑K selection theo budget, densify tới 20k.
Các vấn đề chính còn lại (nguyên nhân hợp lý cho chất lượng thấp)
1. Score chưa phản ánh đúng “contribution” của Gaussian trong render
    * Hiện score dựa trên geometry footprint + δI.
    * Không dùng transmittance/alpha contribution → ranking có thể sai, nhất là cảnh nhiều che khuất.
    * Đây là mismatch lớn nhất so với “adjoint of renderer”.
2. Chưa tách clone/split theo budget riêng tối ưu
    * Dù đã tách score, clone/split vẫn dùng threshold chung theo top‑K ratio.
    * Có thể clone chiếm phần lớn budget → split không đủ → thiếu chi tiết.
3. Tile approximation còn thô
    * Dùng tile‑center sampling, không integrate thật trên tile.
    * Nếu Gaussian nhỏ/edge‑heavy, approximation có thể sai.
4. Densification rate vẫn không ổn định
    * Log vẫn xuất hiện nhiều sel=0 hoặc valid cực thấp.
    * Nghĩa là score phân bố vẫn “zero‑inflated” → ranking không ổn.
5. Overhead tăng ~24%
    * FW mean + var kernels ~2.48ms / 10.3ms core → ~24% overhead.
    * Không gây sai kết quả nhưng làm training chậm hơn baseline.
Kết luận
* Implementation “đúng skeleton” và đã đúng theo SRM proxy ở mức kernel‑average.
* Nhưng vẫn thiếu yếu tố contribution/visibility và budget allocation clone/split, nên ranking densify chưa tốt → chất lượng thua baseline. “”” 
Tôi cần bạn review kĩ lại các bài SOTA khác nhau như Revising Densification / ConeGS / PixelGS / AbsGS, để tham khảo cơ chế của họ và tìm hiểu giải thích được “Tại sao kết quả họ tốt hơn còn tôi chưa tốt bằng 3DGS baseline?”. Tìm hiểu và giải thích được dúngd vấn đề Vậy tại sao method của tôi “đúng-theory-looking” nhưng vẫn thua baseline. Tôi đã thiếu những gì, ví dụ occlusion gating tôi thấy không đúng, trong bài revising họ sử dụng weight visibility. Tôi muốn đảm bảo giữ được story và method của tôi novel nhưng vẫn nâng cấp dựa trên những gì người ta đã chứng minh trước đó. Tập trung nhận định Điểm khiến kết quả tôi không tốt và mismatch cốt lõi giữa theory và “densification trong 3DGS thực tế” đã giết performance của tôi. :a,f sao sửa lại Sửa thế nào vừa đúng “math” vừa novel đảm bảo thuyết phục reviewer.

* * *

**ChatGPT:**

Dựa trên report của bạn (FW densification mean/variance đang **tụt rất sâu** so với baseline 3DGS) thì mình nghĩ vấn đề không phải “ý tưởng SRM sai”, mà là **bạn đang ước lượng sai đại lượng mà SRM thật sự yêu cầu trong setting NVS + alpha‑compositing**, cộng thêm **xấp xỉ tile‑center làm bias cực mạnh** (đặc biệt khi Gaussian nhỏ). Nói kiểu reviewer: _paper có vẻ “theory-looking”, nhưng implementation hiện tại không bám đúng “adjoint của renderer”, nên densification ranking bị sai → densify sai chỗ → chất lượng rơi tự do._

Mình sẽ làm 4 việc:

1. Tóm tắt chính xác “angle” và cơ chế densification của **Revising Densification / Pixel‑GS / AbsGS / ConeGS** (cái họ thực sự sửa).
2. Chỉ ra **mismatch cốt lõi** giữa “SRM theorem” và “densification thực tế trong 3DGS” đang giết performance của bạn.
3. Đề xuất một bản **sửa math‑đúng** (adjoint/visibility‑consistent) nhưng vẫn giữ được story “SRM‑guided” và có điểm novel đủ thuyết phục reviewer.
4. Checklist những test/ablation để chứng minh bạn đã sửa đúng nguyên nhân (không tự lừa mình).

---

## 1) SOTA đang làm gì trong densification và vì sao họ mạnh hơn

### (A) Revising Densification in Gaussian Splatting (ECCV 2024)

**Vấn đề họ nhắm:** ADC của 3DGS “split/clone sai hoặc thiếu”, dẫn tới **localized under‑fitting** ở vùng texture cao (cỏ, lá…) và khó tune threshold vì score dựa trên grad. [arXiv](https://arxiv.org/html/2404.06109v1)

**Key idea:** thay vì dùng “positional gradient magnitude” làm score, họ dùng **per‑pixel error map** (ví dụ SSIM per pixel), rồi **phân bổ (redistribute) lỗi pixel về từng Gaussian theo tỷ lệ đóng góp của Gaussian vào pixel đó** (visibility/contribution weighting). Điểm cốt lõi: _pixel error là “observed”, nhưng phải “credit assignment” đúng cho Gaussian nào thật sự tạo ra pixel đó._ [arXiv](https://arxiv.org/html/2404.06109v1)

**Implementation trick rất quan trọng:** họ không lưu per‑pixel contribution map; họ gắn **một scalar phụ** cho mỗi Gaussian, render scalar đó, rồi thêm **auxiliary loss** dạng dot product giữa error (detach) và ảnh scalar render; gradient w.r.t scalar sẽ ra đúng per‑Gaussian error score. [arXiv](https://arxiv.org/html/2404.06109v1)
→ Đây là một “chiêu” cực hay: **dùng chính backward của renderer để làm assignment**, nên tự động đúng với alpha‑compositing & visibility.

**Bổ sung họ làm để ổn định:**

* **Opacity correction sau cloning:** giữ nguyên opacity khi clone tạo bias (vì alpha‑compositing làm thay đổi weight của phần phía sau). Họ đề xuất giảm opacity sau clone theo nghiệm đóng dạng để giảm bias. [arXiv](https://arxiv.org/html/2404.06109v1)
* Growth control (budget số primitive), opacity regularization; và họ chỉ ra opacity correction có impact lớn trong ablation. [arXiv](https://arxiv.org/html/2404.06109v1)

**Tóm lại angle:** “Đừng dùng gradient heuristic mù mờ; hãy dùng **pixel‑error** + **visibility‑weighted credit assignment** + **opacity‑handling đúng**”.

---

### (B) Pixel‑GS (ECCV 2024)

**Vấn đề họ nhắm:** 3DGS phụ thuộc chất lượng SfM init; nhiều vùng init thưa/thiếu → blur/needle artifacts. Nguyên nhân chính: growth condition của 3DGS dùng **average gradient magnitude qua các view** nên **Gaussian lớn** (thấy ở nhiều view) nhưng mỗi view chỉ “chạm biên” → gradient bị “dilute”, không vượt threshold → **không grow đủ**. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

**Key idea:** thay average theo view bằng **weighted average theo số pixel mà Gaussian thực sự cover ở mỗi view**. Nói thẳng: “đếm pixel” để chống bias do view averaging. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

**Cực kỳ liên quan tới bạn:** họ định nghĩa “pixel tham gia” với điều kiện có **transmittance** đủ lớn và alpha đủ lớn (tức là pixel “thật sự visible, thật sự contribute”), chứ không phải chỉ overlap hình học. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

**Extra:** họ còn scale gradient theo depth để suppress floaters near camera. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

**Angle:** “Growth condition phải phản ánh **pixel coverage & visibility**; không thì bạn sẽ grow sai/thiếu.”

---

### (C) AbsGS (arXiv 2024)

**Vấn đề họ nhắm:** 3DGS bị “over‑reconstruction → Gaussian lớn che vùng texture → blur”, nhưng lại **không split** vì score gradient bị triệt tiêu.

**Cốt lõi họ chứng minh:** view‑space positional gradient của một Gaussian là **tổng các sub‑gradient theo pixel**, và các sub‑gradient này có thể **khác dấu/hướng** (do nhiều term trong alpha‑blending), nên tổng vector có thể nhỏ dù “đáng split” → gọi là **gradient collision**. [ar5iv+1](https://ar5iv.org/html/2404.10484v1)

**Fix:** dùng **homodirectional gradient**: lấy **absolute value** từng thành phần trước khi cộng, để magnitude không bị triệt tiêu bởi direction. [ar5iv](https://ar5iv.org/html/2404.10484v1)
Họ chủ yếu dùng cái này cho **split**; clone vẫn theo baseline. [ar5iv](https://ar5iv.org/html/2404.10484v1)

**Angle:** “Tín hiệu densification phải robust với **cancellation** do hướng gradient; magnitude mới quan trọng.”

---

### (D) ConeGS (2026 / arXiv 2025)

**Vấn đề họ nhắm:** cloning‑based densification phụ thuộc cấu trúc sẵn có; “propagate along existing geometry” → **khám phá kém**, cần rất nhiều primitive để cover hết scene.

**Key idea:** dùng **error map** trong image space, chọn pixel lỗi cao, lấy **depth proxy từ iNGP**, rồi **spawn Gaussian mới dọc theo viewing cone** tại depth đó, size theo đường kính cone (mượn tinh thần Mip‑NeRF). Có thêm opacity penalty và budgeting để remove redundant. [CVlibs](https://www.cvlibs.net/publications/Baranowski2026THREEDV.pdf)

**Angle:** “Đừng chỉ clone từ cái đang có; hãy **spawn mới ở nơi error cao** + depth proxy → placement tốt hơn, đặc biệt khi budget chặt.”

---

## 2) Vì sao method của bạn “đúng‑theory‑looking” nhưng vẫn thua baseline

Bạn đang dùng logic (từ draft của bạn):

* $G(u)=\partial \ell/\partial I(u)$ (image.grad)
* rồi lấy $\int G(u)\,\rho_i(u)\,du$ làm densification signal.

Và bạn **đã tự cảnh báo**: “alpha compositing & visibility → kernel hiệu dụng có transmittance; mình treat $\rho_i$ như proxy footprint và dùng occlusion gating”.
Nhưng vấn đề là: **đoạn “proxy” này đang quá sai trong thực tế**, nên ranking chết.

### 2.1 Mismatch lý thuyết quan trọng nhất: trong SRM, δF cho inverse problem ≠ δI trên ảnh

Trong SRM Theorem 1, gradient w.r.t output vector $v$ có dạng tích phân của **first variation** $\delta F$ với density của splat.
N

Method

roblems**, SRM nói rất rõ: $\delta F[f] = (D_f A)^*(A[f]-g)$, tức là **residual đã được backproject qua adjoint của forward operator**.

Method

ũng mô tả forward operator là RTE và thực tế dùng α‑blending; gradient phụ thuộc xấp xỉ và **tính bằng autodiff**.

=> **Điểm chết người:** bạn đang lấy $G(u)=\partial \ell/\partial I(u)$ làm “δF”, nhưng trong NVS, “δF đúng” phải là **$ (D A)^* G$** (ít nhất theo linearization). Nói ngắn

Splat Regression Models

residual”; thiếu hoàn toàn “adjoint of renderer”**.

Và trong α‑compositing, adjoint này chứa đúng những thứ bạn đang thiếu:

* **transmittance/visibility**,
* **alpha contribution**,
* thậm chí các term phụ thuộc color mixing (AbsGS chỉ ra sub‑gradient sign phụ thuộc nhiều term). [ar5iv+1](https://ar5iv.org/html/2404.10484v1)

Baseline 3DG

Splat Regression Models

ean2D (và các gradient renderer‑provided) — dù heuristic — nhưng nó vẫn “đúng adjoint hơn” vì nó là backprop thật sự qua renderer.

### 2.2 “Footprint overlap” ≠ “contribution”: bạn đang credit assignment sai

Bạn đang assign lỗi theo **geometry footprint**: pixel nào nằm trong bbox/tile rect của Gaussian thì Gaussian “nhận lỗi”.

Nhưng Revising Densification chỉ ra muốn densify đúng phải **redistribute per‑pixel error về Gaussian theo tỷ lệ đóng góp của Gaussian vào pixel**, không phải theo overlap hình học. [arXiv](https://arxiv.org/html/2404.06109v1)

Pixel‑GS cũng định nghĩa “pixel tham gia” dựa trên **transmittance** và alpha threshold, tức là pixel phải **visible** mới nên ảnh hưởng score. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

=> Nếu scene nhiều che khuất, cách của bạn sẽ:

* bơm score vào Gaussian **nằm sau** (occluded) chỉ vì bbox overlap,
* trong khi Gaussian “đang thật sự render pixel đó” lại không được ưu tiên đúng mức.

Occlusion gating bằng depth buffer dạng sigmoid thường **không đủ** vì:

* transmittance là tích luỹ theo thứ tự front‑to‑back (phụ thuộc toàn bộ stack), không phải chỉ “độ lệch depth so với depth map”;
* depth map trong 3DGS thường là depth expectation/alpha‑weighted, không phải “first hit” chuẩn, nên gating rất dễ sai.

### 2.3 Tile‑center approximation có thể đang làm bạn “mù” với Gaussian nhỏ (zero‑inflated score)

Bạn nói log có nhiều `sel=0` và score distribution zero‑inflated.

Một nguyên nhân rất thực dụng (và mình thấy cực hợp với triệu chứng của bạn) là:

* Bạn weight theo conic tại **tile center**. Gaussian nhỏ (σ≈1–2 px) mà tile 16×16: tile center thường cách mean vài px → $e^{-0.5 d^T C d}$ cực bé → underflow → weight ≈ 0 → score ≈ 0.
* Kết quả: densification “nhìn không thấy” những Gaussian nhỏ/edge‑heavy, trong khi chính chúng là thứ cần clone/split để tăng chi tiết.

SOTA không gặp vấn đề này vì:

* họ làm assignment / gradient theo **pixel-level contribution** (Revising/AbsGS) hoặc ít nhất “pixel count” dựa trên pixel coverage thật (Pixel‑GS). [arXiv+2ar5iv+2](https://arxiv.org/html/2404.06109v1)

---

## 3) Sửa thế nào để vừa đúng math vừa giữ story SRM và vẫn novel

Mục tiêu: **giữ “SRM‑guided moment”** nhưng thay “proxy footprint” bằng **renderer‑consistent effective kernel** (visibility / contribution‑aware). Đây là cách sửa đúng nhất để bạn không bị reviewer bắt “you ignored the adjoint”.

### 3.1 Công thức “đúng hướng” (adjoint‑consistent) bạn nên nhắm tới

SRM cho ta dạng tổng quát:

$$\nabla_{v_i} \; F(f_\mu)
\;\propto\; \int \delta F(x)\,\rho_i(x)\,dx.$$

Trong inverse problem, $\delta F$ là backprojected residual qua adjoint: $\delta F = (DA)^*(A[f]-g)$.

Trong 3DGS rasterizer với alpha‑blending, với mỗi pixel $u$, ta có dạng front‑to‑back:

* $$T_0(u)=1$$
* mỗi Gaussian theo thứ tự depth có alpha $\alpha_i(u)$
* đóng góp “màu” của Gaussian i vào pixel u có trọng số (scalar) dạng: $$w_i(u) = T_{i-1}(u)\,\alpha_i(u)$$ (đây chính là “contribution weight”/visibility weight trong nhiều cách nhìn).

Và gradient của loss wrt “color coefficient” (đặc biệt DC SH) về bản chất là sum của $G(u)$ nhân $w_i(u)$. (Ý này cũng là lý do Revising có thể red

Splat Regression Models

contribution to rendered pixel color”.) [arXiv](https://arxiv.org/html/2404.06109v1)

**Sửa cốt lõi:** thay vì dùng

$$m_i = \int G(u)\,\rho_i(u)\,du$$

bạn cần dùng (xấp xỉ):

$$m_i \approx \frac{\sum_u w_i(u)\,G(u)}{\sum_u w_i(u)}.$$

Và “variance” (để split) nên là moment bậc 2 theo đúng weight đó:

$$\mathrm{var}_i \approx
\frac{\sum_u w_i(u)\,\|G(u)\|^2}{\sum_u w_i(u)}
\;-\;
\left\|\frac{\sum_u w_i(u)\,G(u)}{\sum_u w_i(u)}\right\|^2.$$

Cái này **vừa giữ story SRM‑moment**, vừa align với:

* Revising: per‑pixel error → per‑Gaussian score bằng contribution weights [arXiv](https://arxiv.org/html/2404.06109v1)
* Pixel‑GS: “pixel tham gia” phải visible (transmittance) [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
* AbsGS: cancellation do direction → nên dùng moment/abs/energy thay vì vector sum thuần [ar5iv](https://ar5iv.org/html/2404.10484v1)

### 3.2 Cách implement “Plan B CUDA-kernel” mà vẫn bám theory (không phá wall‑clock)

Bạn đang có sẵn:

* tile‑reduce $G(u)$ → $s_t$ và $q_t$
* binningBuffer/geomBuffer để biết list Gaussian theo tile và depth order (giống rasterizer)

**Patch đề xuất (rất đáng làm):** thay “Gaussian‑centric sum over tiles với conic weight thuần” bằng **tile‑centric compositing tại tile samples** để có **transmittance‑aware weight**.

#### (i) Tile moments như bạn đã làm

Giữ:

* $$s_t = \frac{1}{|\Omega_t|}\sum_{u\in\Omega_t} G(u)$$
* $$q_t = \frac{1}{|\Omega_t|}\sum_{u\in\Omega_t} \|G(u)\|^2$$

(đây là Eq. (5)(6) trong draft bạn).

#### (ii) Tính $w_{i,t}$ có transmittance bằng “low-res alpha compositing” trên tile

Với mỗi tile $t$, lấy 1–4 sample point $u_{t,s}$ (center + corners là đủ để giảm bias tile‑center).
Duyệt danh sách Gaussian trong tile **theo depth order** (binningBuffer đã có).

Pseudo:

* $$T \leftarrow 1$$
* for gaussian i in tile t (sorted front→back):
    * compute $\alpha_{i}(u_{t,s})$ theo đúng rasterizer (opacity × exp(power) và clamp)
    * average alpha qua samples: $\bar\alpha_{i,t} = \frac{1}{S}\sum_s \alpha_i(u_{t,s})$
    * weight “visibility/contribution” của i ở tile: $$w_{i,t} = T \cdot \bar\alpha_{i,t}$$
    * update $T \leftarrow T(1-\bar\alpha_{i,t})$
    * atomic accumulate:
        * `sum_w[i] += w_{i,t}`
        * `sum_wg[i] += w_{i,t} * s_t` (vector3)
        * `sum_wg2[i] += w_{i,

            Method

            au đó:

$$m_i = \frac{\text{sum\_wg}_i}{\text{sum\_w}_i},\quad
\mathrm{var}_i =
\frac{\text{sum\_wg2}_i}{\text{sum\_w}_i} - \|m_i\|^2.$$

**Tại sao cái này “đúng theory hơn” implementation hiện tại của bạn?**
Vì bạn đã chuyển từ “footprint proxy” sang **ước lượng trực tiếp $w_i(u)=T\alpha$** (adjoint/visibility) trong alpha‑compositing — đúng thứ mà SRM nói bạn phải đưa vào δF ở inverse problem (adjoint).

**Tại sao nó có khả năng cứu PSNR/SSIM/LPIPS?**
Vì nó gần như “revising‑style credit assignment” nhưng bạn dùng $G=\partial \ell/\partial I$ và còn có moment bậc 2. Revising thắng vì họ assignment đúng; bạn thua vì assignment sai. [arXiv](https://arxiv.org/html/2404.06109v1)

#### (iii) Clone vs split — giữ “moment story” nhưng làm giống cộng đồng

* Clone score: $S^\text{clone}_i = \|m_i\|$ (hoặc $\|m_i\|_1$ để giảm cancellation màu)
* Split score: $S^\text{split}_i = \mathrm{var}_i \cdot \psi(r_i)$, với $\psi(r)$ tăng theo radius để ưu tiên Gaussian lớn (giống trực giác AbsGS: split large Gaussians ở vùng detail). [ar5iv](https://ar5iv.org/html/2404.10484v1)

#### (iv) Liên hệ với Pixel‑GS để “sạch story”

Pixel‑GS nói muốn chống bias view‑averaging thì dùng “#pixels covered” làm weight. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
Trong pipeline mới, **$\text{sum\_w}_i$** chính là một proxy rất tự nhiên cho “covered & visible area” (vì nó là tổng $T\alpha$).
=> Khi bạn accumulate score qua nhiều iteration/view, bạn có thể:

* average $m_i$ theo weight sum_w thay vì average đều,
* hoặc chỉ dùng sum_w để reject những view mà Gaussian barely visible.

Điều này vừa đúng “community direction”, vừa không phá story SRM.

---

## 4) Reviewer sẽ hỏi gì, và bạn nên “bịt lỗ” story thế nào để vẫn novel

Nếu mình là reviewer, mình sẽ hỏi (và đây là điểm bạn phải chủ động trả lời trong paper):

### Q1: “SRM Theorem 1 dùng δF; tại sao bạn dùng δI?”

Bạn phải trả lời theo hướng **Example 2 của SRM**: trong inverse problems, δF là **adjoint backprojection**; còn δI chỉ là gradient ở output.

**Câu trả lời đúng hướng:**

* “Chúng tôi không coi δI là δF cho NVS nguyên bản; thay vào đó, chúng tôi xây một proxy cho $(DA)^* \delta I$ dưới alpha‑blending bằng cách đưa vào transmittance/visibility weight $w=T\alpha$.”

=> Đây là chỗ bạn “upgrade” method để khỏi bị bắt lỗi “theory misuse”.

### Q2: “Cái này có giống Revising Densification không?”

Giống ở **credit assignment theo contribution**, và bạn nên thừa nhận nó là nền móng vững: Revising làm đúng chỗ mà ADC baseline sai. [arXiv+1](https://arxiv.org/html/2404.06109v1)

**Novelty bạn có thể bán:**

* Revising dùng **pixel error scalar** và chủ yếu làm score 0th‑order; bạn dùng **backprop error signal $G=\partial \ell/\partial I$** + **moment bậc 1 và bậc 2** để:
    * tách clone vs split “mang tính cấu trúc” (variance)
    * liên hệ trực tiếp với SRM gradient integral form (Theorem 1) và inverse-problem adjoint (Example 2).
* Và bạn có “efficient tile‑moment implementation” (một đóng góp systems‑CV rõ ràng).

### Q3: “Cái variance của bạn có thực sự liên quan AbsGS không?”

Bạn trích AbsGS đúng cách: họ chỉ ra cancellation do direction và họ lấy abs để tránh collision. [ar5iv](https://ar5iv.org/html/2404.10484v1)
Variance/energy là một cách khác để đo “không đồng nhất tín hiệu trong footprint”, cũng chống cancellation. Bạn nên làm ablation:

* split dùng AbsGS abs‑gradient vs split dùng variance moment,
* hoặc kết hợp: split score = energy (E[||G||^2]) thay vì var.

### Q4: “Occlusion gating kiểu depth buffer có đáng tin không?”

Bạn nên bỏ “depth‑sigmoid gating” như điểm trụ story. Reviewer sẽ nói “depth buffer không phải visibility trong alpha compositing”.
Thay vào đó, hãy nói: “visibility weight xuất phát từ transmittance trong alpha compositing” — đúng hướng Revising/Pixel‑GS.

---

## 5) Tại sao SOTA của họ tốt hơn còn bạn thua baseline: câu trả lời thẳng, không vòng vo

* **Revising** tốt hơn vì họ giải quyết đúng bài toán _credit assignment_: pixel error phải đổ về đúng Gaussian đã contribute pixel đó; bạn đang đổ theo footprint nên sai nặng khi có occlusion. [arXiv](https://arxiv.org/html/2404.06109v1)
* **Pixel‑GS** tốt hơn vì họ sửa bias của growth condition bằng pixel coverage “visible”, có transmittance threshold; cách của bạn tile‑center + footprint có thể “không thấy” pixel coverage thật. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
* **AbsGS** tốt hơn vì họ phân tích đúng cơ chế triệt tiêu gradient (gradient collision) và đưa ra estimator robust (abs) — còn bạn lấy mean vector của δI dễ bị cancellation mà lại thiếu các term compositing. [ar5iv+1](https://ar5iv.org/html/2404.10484v1)
* **ConeGS** tốt hơn (đặc biệt dưới budget thấp) vì họ không bị giới hạn “clone dựa trên structure hiện tại”; họ đặt primitive mới trực tiếp theo pixel error + depth proxy. [CVlibs](https://www.cvlibs.net/publications/Baranowski2026THREEDV.pdf)

Còn bạn thua baseline vì baseline tuy heuristic nhưng ít nhất score của nó “đi cùng renderer backprop”, còn score của bạn hiện tại đang là “footprint‑only proxy” nên lệch khỏi adjoint → densify sai → tụt.

---

## 6) Checklist ablation để chứng minh bạn đã sửa đúng “mismatch cốt lõi”

Sau khi bạn làm patch “tile‑center transmittance weight” (w=Tα) như mục 3, mình khuyên chạy đúng các ablation này để reviewer tin:

1. **Footprint‑only vs Visibility‑weighted (w=Tα)**
    * Giữ mọi thứ giống hệt (budget, schedule, threshold).
    * Nếu visibility‑weighted không kéo metrics lên mạnh thì có bug ở alpha computation / depth order / T update.
2. **Tile center 1 sample vs 4 samples (center+corners)**
    * Nếu 4 samples giảm zero‑inflation và tăng densify stability → xác nhận tile‑center bias là thật.
3. **Split score: var vs energy vs AbsGS‑style abs**
    * var = E[||G||^2] − ||E[G]||^2
    * energy = E[||G||^2]
    * abs‑mean = ||E[|G|]||
        Kết quả nào tốt hơn sẽ cho bạn “story” rõ ràng: bạn đang chống cancellation kiểu nào.
4. **Clone/split budget tách riêng**
    * 50/50, 30/70, schedule theo iteration.
    * Bạn sẽ thấy ngay nếu “split bị đói” là nguyên nhân blur.
5. **So sánh trực tiếp với Revising score**
    * Nếu bạn implement được Revising auxiliary score (per‑pixel error assignment) thì compare overlap của top‑K selected Gaussian giữa 2 score.
    * Nếu overlap thấp nhưng Revising thắng → bạn vẫn missing something trong w hoặc moment.

---

## Kết luận “định hướng sửa” (mình chọn một hướng mạnh nhất để bạn theo)

Nếu bạn muốn **giữ story SRM** mà vẫn “đúng khoa học & đúng thực tế 3DGS”, thì hướng có giá trị nhất là:

> **Biến SRM moment integral thành “renderer‑adjoint consistent”: thay footprint proxy bằng visibility/contribution weight (transmittance × alpha) và tính moment bậc 1/2 theo weight đó.**

Nó:

* giải đúng mismatch mà SRM Example 2 cảnh báo (inverse problem cần adjoint),
* bám sát cộng đồng (Revising/Pixel‑GS đều nhấn mạnh visibility/contribution), [arXiv+2ECVA+2](https://arxiv.org/html/2404.06109v1)
* vẫn giữ novelty (moment‑based clone/split + efficient tile implementation + liên hệ SRM Theorem 1),
* và quan trọng nhất: **khả năng rất cao sẽ kéo performance lên lại baseline hoặc vượt**, vì bạn đang sửa đúng “credit assignment”.

Nếu bạn muốn, mình có thể giúp bạn draft lại phần “Method”/“Theory” để update story từ “footprint proxy” sang “adjoint‑consistent effective kernel” (giữ nguyên paper skeleton của bạn), nhưng ở đây mình đã cố tình tập trung vào thứ sẽ cứu kết quả và cứu reviewer‑logic trước tiên.

* * *

**You:**

Đây là report của tôi, dựa trên đề xuất của bạn, bạn có nghĩ điều chỉnh như bạn đề xuất có giải quyết đúng vấn đề chúng đang gặp phải không. Hãy kiểm tra lại bài toán chi tiết và verify lại theory bạn vừa đề xuất có đúng với 3dgs densification không. Liệu hướng bạn có đưa kết quả vượt baseline 3dgs không dựa trên các bài SOTA gần đây, Tôi muốn tiếp tục giữ story SRM mà vẫn “đúng khoa học & đúng thực tế 3DGS”, liệu hướng của bạn có giá trị trong cộng đồng không. Phân tích lại logic và review lại tập trung những phần quan trọng, nếu đúng thì hãy trình bày lại method đảm bảo vẫn “đúng khoa học & đúng thực tế 3DGS”. Đảm bảo đang phân tích đúng nguyên nhân và có cơ sở để có performance tốt, kiểm tra kết quả sanitiy. Formalize lại method draft đảm bảo giải quyết đúng các rủi ro và kì vọng. “”” Báo cáo ngắn – Kết quả FW Densification (Mean/Variance) Kết quả tổng quan * PSNR/SSIM/LPIPS vẫn thấp hơn baseline: * Ours @30k: PSNR 20.92, SSIM 0.54, LPIPS 0.45 * Baseline ~25.22 / 0.76 / 0.21 Sanity check (fw_sanity.json) * spearman_fw_vs_featgrad = 0.26 * spearman_mean2d_vs_featgrad = 0.65 → FW score tương quan kém hơn mean2D với gradient thật ⇒ chưa đúng “theory behavior”. Oracle test (fw_oracle.json) * corr(score, gain): * FW = 0.166 * mean2D = -0.026 * |I−GT| = 0.607 (cao nhất) → FW score không dự đoán gain tốt, abs‑error proxy tốt hơn. Ablation (fw_ablation.json) * corr_fw_featgrad = 0.17, corr_mean2d_featgrad = 0.70 * corr_gated_fw_featgrad = 0.70 → Gating giúp FW ngang mean2D ⇒ issue chính là contribution mismatch / occlusion. Overhead (fw_overhead.json) * FW overhead ~ 8.3% core time (chấp nhận được). → Chậm không phải nguyên nhân chính làm kém chất lượng. Budget curve (budget_curve.json) * 7k → PSNR ~19.9 * 30k → PSNR ~20.9 → Chất lượng tăng rất ít dù số Gaussians tăng ⇒ score chọn densify không hiệu quả. Kết luận * Implementation skeleton đúng (δI, tile-sum, footprint) * Nhưng thực nghiệm cho thấy score không đúng hành vi theory (không tương quan gain). * Dữ liệu chỉ ra contribution/occlusion mismatch là nguyên nhân chính (gated FW cải thiện).  “””

* * *

**You:**

ưu tiên theo những phương pháp SOTA Gần đây và phân tích hướng đi dựa trên những bài SOTA có kết quả tốt nhất vì kết qủa tốt là minh chứng cho phương pháp tốt. Đọc kết quả để đưa ra hướng đi tốt nhất đảm bảo nhưng vẫn theo câu chuyên theory trong bài báo SRM tôi cung cấp

* * *

**ChatGPT:**

Got it – I’ve updated the task according to your latest instructions and am on it.

* * *

**ChatGPT:**

Dựa trên report của bạn, mình nghĩ vấn đề “FW densification nhìn đúng-theory nhưng thua baseline nặng” **không phải** do bạn implement sai skeleton (δI, tile-sum, footprint), mà do một **mismatch cốt lõi giữa cái SRM Theorem 1 nói và cái renderer 3DGS thật sự làm** — cụ thể là **alpha compositing + visibility/transmittance**. Bạn đã tự “bắt quả tang” nó bằng ablation: _gated FW_ kéo Spearman lên ngang mean2D, tức là lỗi chính nằm ở **contribution attribution**, không phải overhead hay noise.

Mình sẽ làm 3 việc:

1. Review nhanh các SOTA bạn nêu (Revising Densification / Pixel-GS / AbsGS / ConeGS) theo đúng “cơ chế densification + vì sao họ win”.
2. Formalize lại mismatch “SRM vs 3DGS thực tế” bằng toán (adjoint của renderer).
3. Đề xuất hướng **có cơ sở nhất** để vừa giữ story SRM vừa đúng thực tế 3DGS và có cửa vượt baseline, kèm method draft “paper-ready” và checklist triển khai/sanity.

---

## 1) Chẩn đoán kiểu reviewer từ số liệu bạn đưa

Các con số của bạn nói rất thẳng:

* **FW score hiện tại không giống hành vi gradient thật**:
    * `spearman_fw_vs_featgrad = 0.26` trong khi `mean2d_vs_featgrad = 0.65` → ranking của FW đang sai hướng.
    * Oracle test: `corr(score, gain)` của FW = 0.166, trong khi `|I−GT|` = 0.607 → “độ lỗi ảnh” dự đoán gain tốt hơn nhiều.
* **Khi thêm gating thì FW lên ngang baseline**:
    * `corr_gated_fw_featgrad = 0.70` ~ `corr_mean2d_featgrad = 0.70`
        → _Nếu attribution/visibility được xử lý đúng_, FW không tệ về mặt “có tín hiệu”.
* **Tăng số Gaussian nhưng PSNR tăng rất ít** (`7k→19.9`, `30k→20.9`)
    → densify đang “đốt budget” vào chỗ không đem lại cải thiện (ranking sai, hoặc clone/split allocation sai, hoặc signal sai).

Kết luận reviewer-style: _Bạn không thiếu “theory-looking”, bạn thiếu “adjoint-correctness” (đúng theo đạo hàm của renderer)_.

---

## 2) SOTA gần đây làm gì và vì sao họ tốt hơn bạn

### 2.1 Revising Densification in Gaussian Splatting (Rota Bulò et al., 2024)

**Góc nhìn:** “Densification nên được **guidance bởi per-pixel error** chứ không phải positional gradient.” [ar5iv](https://ar5iv.org/pdf/2404.06109)

**Điểm mấu chốt thắng:**

1. **Error attribution đúng theo alpha compositing**
    Họ định nghĩa error cho từng primitive bằng cách “chia” per-pixel error về primitive **tỉ lệ theo hệ số alpha-compositing** (tức là contribution thật sự của Gaussian vào pixel, đã bao gồm visibility/transmittance). [ar5iv](https://ar5iv.org/pdf/2404.06109)
    → Đây chính là thứ bạn thiếu khi chỉ dùng footprint + δI.
2. **Fix bias từ cloning opacity**
    Cloning “naive” làm thống kê alpha-blending bị lệch (vì (1−o)^2 ≠ (1−o)), họ sửa bằng opacity correction khi clone. [ar5iv](https://ar5iv.org/pdf/2404.06109)
    → Nếu bạn clone nhiều hơn baseline, bias này giết chất lượng nhanh.
3. **Evidence mạnh về failure của gradient-based densification**
    Họ chỉ ra có cảnh mà GS densification logic “không densify đúng vùng” **kể cả đẩy lên 10M primitives** vẫn blur, vì gradient-based score có thể “tụt cực thấp” ở vùng underrepresented. [ar5iv](https://ar5iv.org/pdf/2404.06109)
    → Điều này trùng vibe với budget curve của bạn: tăng N mà chất lượng không lên.

**Tóm tắt:** Họ thắng vì **attribution đúng (alpha/visibility)** + **signal tốt (error)** + **sửa clone bias**.

---

### 2.2 Pixel-GS (Zhang et al., ECCV 2024)

**Góc nhìn:** vẫn dùng gradient-based densification, nhưng sửa 2 vấn đề:

1. **Pixel-aware gradient**: weighting gradient theo số pixel Gaussian thật sự cover ở mỗi view (tránh view “đụng vài pixel” nhưng gradient nhiễu). Công thức họ dùng là trung bình gradient magnitude qua views nhưng weight bởi pixel count normalized. [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
2. **Visibility / contribution filter rõ ràng**
    Họ định nghĩa pixel “tham gia” bằng điều kiện transmittance và alpha ngưỡng (đảm bảo pixel đó thật sự thấy Gaussian chứ không bị occlude). [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
    → Đây chính là dạng “occlusion gating” nhưng _chuẩn hơn_ (theo T tích luỹ trong alpha-blending).
3. **Scaled gradient field để giảm floaters**
    Pixel-aware gradient “thô” có thể tạo floaters ở scene lớn; họ scale gradient để kiểm soát densify. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

**Kết quả**: cải thiện PSNR/SSIM và đặc biệt LPIPS so với 3DGS baseline trên Mip-NeRF360/Tanks&Temples. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)

**Tóm tắt:** Pixel-GS thắng vì **đúng visibility hơn** + **giảm view-noise** + **chống floaters**.

---

### 2.3 AbsGS (Ye et al., 2024)

**Góc nhìn:** “Gradient magnitude của 3DGS bị lỗi _gradient collision_.”

**Điểm mấu chốt:**

* Positional gradient là tổng của nhiều pixel-wise gradients; hướng khác nhau → triệt tiêu → gradient nhỏ giả, làm split/cloning không trigger đúng. Họ phân tích và chứng minh bằng decomposition theo pixel. [ar5iv](https://ar5iv.org/pdf/2404.10484)
* Fix: **homodirectional gradient**: lấy trị tuyệt đối từng thành phần trước khi cộng để tránh cancellation, rồi dùng nó làm guidance cho split. [ar5iv](https://ar5iv.org/pdf/2404.10484)
* Kết quả: cải thiện SSIM/LPIPS, đôi khi giảm memory. [ar5iv](https://ar5iv.org/pdf/2404.10484)

**Tóm tắt:** AbsGS thắng vì **cắt đúng failure mode “cancellation”** trong gradient-based trigger.

---

### 2.4 ConeGS (2026, hướng khác hẳn)

**Góc nhìn:** “Clone/split bị giới hạn trong việc khám phá vùng chưa có geometry tốt. Hãy densify theo error bằng cách spawn Gaussian theo pixel cone.” [arXiv](https://arxiv.org/abs/2511.06810)
Cái này nhảy sang “birth new Gaussians ở nơi error cao”, thường cần depth estimate/strategy khác → có thể mạnh, nhưng nó không còn đúng “drop-in replacement densification trigger” như bạn đang làm.

---

## 3) Mismatch cốt lõi: SRM Theorem 1 ≠ footprint-only trong 3DGS

### 3.1 SRM nói gì (đúng phần bạn bám)

SRM Theorem 1 (Wasserstein gradient) cho thành phần theo output vector $v$:

$$\nabla_v F(f_\mu)(v,A,b) \;=\; \int \delta F(x)\,\rho_{A,b}(x)\,\pi(dx)$$

và còn có gradient theo $b,A$ có dạng tích phân với $\langle \delta F(x),v\rangle$ và $\nabla \log\rho$.

Trong Method draft của bạn, bạn dùng đúng tinh thần: lấy $G(u)=\partial \ell/\partial I(u)$ rồi “kernel-average” trên footprint.

### 3.2 Nhưng 3DGS renderer thật sự là alpha compositing (adjoint kernel khác!)

3DGS không phải “f(x)=∑ v_i ρ_i(x)” đơn giản. Nó là:

* project Gaussian → 2D gaussian $g_i(u)$
* alpha blending front-to-back: pixel color là tổng weighted bởi **alpha-compositing coefficient

    Splat Regression Models

    i(u),\sigma(o_i),g_i(u)) (trong đó $T_i$ là transmittance trước i).

Và đây là điểm chết người:

Gradient đúng theo chain rule là:

Method

{\partial v_i}
= \sum_u \frac{\partial L}{\partial I(u)} \cdot \frac{\partial I(u)}{\partial v_i}
]
Với alpha compositing, $\partial I(u)/\partial v_i$ **không phải** chỉ là footprint $g_i(u)$. Nó bị nhân bởi **$\alpha_i(u)$ hoặc ít nhất bởi visibility/transmittance** (tuỳ bạn đang xem v_i là màu, SH coeff hay scalar attribute).

Nói dân dã: pixel gradient $\delta I(u)$ chỉ nên “đổ blame” về Gaussian **theo mức Gaussian đó thực sự đóng góp vào pixel**. Nếu Gaussian bị che khuất, $\alpha_i(u)\approx 0$ → blame phải ≈0.

Cái bạn đang làm hiện tại là:

* chỉ dùng $g_i(u)$ (footprint) _không có_ transmittance
    → Gaussian bị occlude vẫn ăn điểm cao nếu nằm trong vùng error mạnh → ranking sai.

Chính SRM paper cũng nhấn mạnh ở bài toán NVS: forward model dùng α-blending là một approximation của RTE và gradient phụ thuộc vào approximation cụ thể và thực tế thường tính bằng autodiff.
Điều này “bảo kê” cho việc: muốn SRM-story đúng trong 3DGS, bạn phải respect **renderer adjoint**, chứ không phải footprint đơn thuần.

### 3.3 Bạn đã verify mismatch này bằng thực nghiệm

* FW vs featgrad Spearman thấp (0.26) → attribution sai.
* “gated FW” lên 0.70 → khi bạn đưa vào chút visibility, ranking trở lại đúng.

Vậy mismatch cốt lõi đã được chốt: **$\rho_i(u)$ trong SRM story của bạn phải được thay bằng “effective

Splat Regression Models

**, ít nhất là một proxy của $\alpha_i(u)$.

---

## 4) Hướng đi “đáng tiền nhất” để vừa giữ SRM story vừa win thực tế (theo SOTA)

Nếu mình phải chọn 1 hướng để tối đa cơ hội vượt baseline _và reviewer A_ nghe hợp lý*, thì mình chọn:

### **Adjoint-Weighted SRM Densification (AW-SRM): error/gradient moment nhưng weight theo alpha/visibility thật sự**

Lý do:

* Nó trực tiếp sửa root cause (contribution mismatch).
* Nó align mạnh với SOTA tốt nhất: Revising Densification + Pixel-GS đều đặt nặng visibility/alpha attribution. [ar5iv+1](https://ar5iv.org/pdf/2404.06109)
* Bạn còn có thể giữ “SRM Theorem 1” làm backbone: vẫn là $\int \delta F \,\rho_{\text{eff}}$. Chỉ khác là $\rho_{\text{eff}}$ phải reflect renderer.

Bên dưới là một draft method “paper-ready” (mình viết theo phong cách formal + implementation-consistent).

---

# Method (draft, SRM-consistent + 3DGS-consistent)

## 4.1 Setup

Cho scene được biểu diễn bởi $N$ Gaussians $\Theta=\{\theta_i\}_{i=1}^N$. Renderer $R(\Theta;c)$ với camera $c$ sinh ảnh RGB:

$$I = R(\Theta;c)\in\mathbb{R}^{3\times H\times W}.$$

Training objective:

$$\min_{\Theta}\; \mathcal{L}(\Theta)=\mathbb{E}_{c\sim\mathcal{C}}\big[\ell(R(\Theta;c), I^{gt}_c)\big].$$

3DGS interleave optimize + densify/prune.

## 4.2 SRM-guided densification signal: từ “footprint-only” sang “renderer-adjoint kernel”

SRM Theorem 1 cho gradient theo output vector $v$ có dạng tích phân của first variation $\delta F$ qua kernel $\rho$.

Trong 3DGS, pixel color là alpha-composited. Với mỗi pixel $u$, ta có (ký hiệu rút gọn):

$$I(u)=\sum_{i\in\mathcal{P}(u)} \alpha_i(u)\,C_i(u)\;+\;T_{bg}(u)\,C_{bg}$$

với $\alpha_i(u)=T_i(u)\,\sigma(o_i)\,g_i(u)$ là alpha-compositing coefficient của Gaussian $i$ tại pixel $u$.

Khi đó, với bất kỳ scalar attribute $s_i$ được alpha-composite giống “một kênh màu” (định nghĩa ở dưới), đạo hàm có dạng:

\frac{\partial I_s(u)}{\partial:contentReference[oaicite:20]{index=20}d\Rightarrow\quad \frac{\partial \mathcal{L}}{\partial s_i} =\sum_{u}\underbrace{\frac{\partial\mathcal{L}}{\partial I_s(u)}}_{\text{pixel signal}}\;\alpha_i(u).

Điểm quan trọng: **kernel hiệu dụng chính là $\alpha_i(u)$**, nó đã bao gồm transmittance/visibility. Đây là thứ mà footprint-only proxy đang thiếu.

## 4.3 Chọn pixel signal: gradient hay error?

Từ report của bạn, “oracle” cho thấy $|I-I^{gt}|$ dự đoán gain tốt hơn FW score. Đồng thời Revising Densification cũng cho thấy error-guided densification outperforms gradient-based trên nhiều scenes. [ar5iv+1](https://ar5iv.org/pdf/2404.06109)

Vì vậy, đề xuất dùng **một guidance error map không âm** $E(u)\ge 0$, ví dụ:

* $E(u)=\|I(u)-I^{gt}(u)\|_1$ (L1 residual)
* hoặc SSIM dissimilarity (như Revising Densification) [ar5iv](https://ar5iv.org/pdf/2404.06109)

Ta **detach** $E$ khỏi graph để nó chỉ dùng cho densification (không bẻ gradient training).

## 4.4 Alpha-weighted moments cho clone/split (thay vì footprint moment)

Ta muốn thống kê error “trong footprint thực sự đóng góp” của Gaussian $i$, tức là weight theo $\alpha_i(u)$.

Định nghĩa các moment (per-view, hoặc accumulate qua window):

$$Z_i=\sum_u \alpha_i(u),\qquad
M_i=\sum_u E(u)\,\alpha_i(u),\qquad
Q_i=\sum_u E(u)^2\,\alpha_i(u).$$

Sau đó:

$$\mu_i=\frac{M_i}{Z_i+\varepsilon},\qquad
\mathrm{Var}_i=\frac{Q_i}{Z_i+\varepsilon}-\mu_i^2.$$

**Score đề xuất:**

* **Clone score** (under-reconstruction / thiếu capacity): $$s^{clone}_i = M_i
    \quad\text{(total attributed error mass)}$$ hoặc $s^{clone}_i=(Z_i+\varepsilon)\mu_i=M_i$ (tương đương).
    Intuition: Gaussian càng “đang chịu blame” nhiều (và thật sự đóng góp) thì clone thêm quanh nó có lợi.
* **Split score** (high-frequency / over-reconstruction / cancellation): $$s^{split}_i = (Z_i+\varepsilon)\,\mathrm{Var}_i\cdot \psi(r_i)
    =\Big(Q_i-\frac{M_i^2}{Z_i+\varepsilon}\Big)\cdot \psi(r_i),$$ với $r_i$ là screen-space radius; $\psi$ monotone (ví dụ $\psi(r)=r$) để ưu tiên split Gaussian footprint lớn.

Điểm hay: split score là “energy của residual không giải thích được bởi mean”, tương tự tinh thần second central moment trong Method draft cũ của bạn và cũng cùng họ với “gradient collision” của AbsGS (tránh cancellation). [ar5iv](https://ar5iv.org/pdf/2404.10484)

## 4.5 Budgeted restricted LMO (Frank–Wolfe story, giữ narrative SRM)

Densification vẫn cần chọn subset (top-K/budget) — reviewer sẽ chấp nhận nếu bạn formalize nó là restricted LMO.

Mỗi densify step:

* chọn top-$K_{clone}$ theo $s^{clone}$ trong nhóm “small gaussians”
* chọn top-$K_{split}$ theo $s^{split}$ trong nhóm “large gaussians”
* global cap $N\le N_{max}$ và prune.

Đây khớp với

Method

cần budget explicit để tránh bursty growth. [ar5iv+1](https://ar5iv.org/pdf/2404.06109)

## 4.6 Opacity correction khi clone

Adopt opacity correction của Revising Densification để tránh bias alpha blending do cloning. [ar5iv](https://ar5iv.org/pdf/2404.06109)
(Trong paper bạn có thể viết dạng “solve equation to keep equivalent transmittance after cloning”.)

## 4.7 Thu được $Z_i,M_i,Q_i$ bằng cách nào (quan trọng cho “implementation-consistent”)?

Đây là “mấu chốt SOTA” và nó giải quyết đúng mismatch của bạn:

**Trick kiểu Revising Densification:** render thêm vài kênh scalar $S^{(k)}(u)$ với per-Gaussian scalar $s_i^{(k)}$, alpha-composite giống RGB:

$$S^{(k)}(u)=\sum_i \alpha_i(u)\, s_i^{(k)}.$$

Tạo auxiliary loss:

$$\mathcal{L}_{aux}
=\sum_u \Big(
1\cdot S^{(0)}(u)\;+\;E(u)\cdot S^{(1)}(u)\;+\;E(u)^2\cdot S^{(2)}(u)
\Big),$$

với $E$ **detach**.

Khi backprop:

$$\frac{\partial \mathcal{L}_{aux}}{\partial s_i^{(0)}}=\sum_u \alpha_i(u)=Z_i,\;
\frac{\partial \mathcal{L}_{aux}}{\partial s_i^{(1)}}=\sum_u E(u)\alpha_i(u)=M_i,\;
\frac{\partial \mathcal{L}_{aux}}{\partial s_i^{(2)}}=\sum_u E(u)^2\alpha_i(u)=Q_i.$$

Vậy bạn lấy đúng moments **mà không cần reconstruct transmittance** — renderer/backward làm hộ. Đây chính là tinh thần “error redistributed proportionally to alpha-compositing coefficients” của Revising Densification. [ar5iv](https://ar5iv.org/pdf/2404.06109)

**Điểm ăn tiền:** đây là cách “đúng khoa học & đúng thực tế 3DGS” nhất, vì nó dùng đúng adjoint của renderer.

---

## 5) Vì sao hướng này có cơ sở để vượt baseline (chứ không phải niềm tin mù)

### 5.1 Nó sửa đúng root cause bạn đo được

Bạn đã thấy: “gated FW” kéo corr lên ngang baseline. Hướng AW-SRM làm mạnh hơn gating:

* không chỉ gate tại mean depth,
* mà attribution per-pixel theo $\alpha_i(u)$ (tức exact visibility/transmittance theo rasterizer).

=> kỳ vọng: ranking tốt hơn mean2D và tốt hơn FW hiện tại.

### 5.2 Nó align với SOTA có kết quả tốt

* Revising Densification: error-guided + alpha attribution + opacity correction → outperforms baseline trên nhiều metrics/scenes (đặc biệt LPIPS). [ar5iv+2ar5iv+2](https://ar5iv.org/pdf/2404.06109)
* Pixel-GS: visibility-aware pixel participation + view weighting → cải thiện PSNR/SSIM/LPIPS so với 3DGS*. [ECVA+2ECVA+2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
* AbsGS: xử lý cancellation → cải thiện chi tiết, SSIM/LPIPS. [ar5iv](https://ar5iv.org/pdf/2404.10484)

Bạn đang fail đúng những thứ họ fix: _visibility attribution_, _cancellation_, _clone bias_. Hướng này gói cả 3.

### 5.3 Nó giữ story SRM mà không “bịa”

* SRM Theorem 1 thật sự nói gradient là tích phân “first variation × kernel”.
* Trong 3DGS, kernel hiệu dụng trong adjoint không phải footprint density mà là **alpha coefficient** (footprint × opacity × transmittance).
* SRM paper cũng nhấn mạnh NVS/α-blending là module và gradient phụ thuộc approximation cụ thể, thường lấy bằng autodiff.

Vậy story không hallucinate: bạn chỉ “sửa kernel cho đúng với forward model thực”.

---

## 6) Rủi ro còn lại và cách “sanitize” để không lại stuck

### Rủi ro A: “Ok score corr cao nhưng vẫn không tăng PSNR/LPIPS”

Cách verify nhanh:

1. Lặp lại oracle test nhưng với score mới:
    * corr(M_i, gain), corr(split_score, gain)
        Nếu vẫn thấp trong khi corr(|I-GT|, gain) cao → c

        Splat Regression Models

        /split action hoặc schedule/budget đang sai, không phải score.
2. Visualize selection: tô trắng top-K gaussians giống AbsGS Fig.4 (bạn đã làm kiểu đó rồi). Nếu score mới chọn đúng vùng error (grass/texture) thì ổn.

### Rủi ro B: Implementation overhead

Cái trick sca

Splat Regression Models

ênh có overhead. Nhưng bạn đang chạy thêm FW kernels ~8% core time; thêm vài kênh scalar trong rasterizer có thể tương đương hoặc thấp hơn tuỳ tối ưu.

Nếu sợ overhead:

* Chỉ render **1 kênh** (M_i) trước; var_i là bonus later.
* Hoặc tính var_i thưa hơn (mỗi Td iter).

### Rủi ro C: “Floaters / over-densify”

Đây là thứ Pixel-GS cảnh báo: densify theo gradient/error có thể tạo floaters nếu không scale theo depth/scene scale. [ECVA+1](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
Giải pháp (nhẹ, implementable):

* Scale score theo depth: ví dụ nhân với $\min(1, (z_i/z_0)^2)$ hoặc clamp theo khoảng cách để không spam gần camera.
* Dùng pruning/opacity decay mạnh hơn cho “newborn” gaussians không cải thiện loss.

---

## 7) Kết luận chọn hướng (đúng tiêu chí “ưu tiên SOTA + giữ SRM”)

Nếu mục tiêu của bạn là **đưa kết quả vượt baseline 3DGS** và vẫn viết được story SRM sạch sẽ, thì:

* **Đừng cố cứu footprint-only FW bằng tuning threshold nữa**. Bạn đã có bằng chứng: nó sai adjoint vì thiếu alpha/visibility.
* Hướng có giá trị nhất là **đưa score về đúng “alpha-weighted attribution”** (Revising Densification / Pixel-GS đều dựa vào điều này) rồi mới thêm “moment/variance” để ăn thêm phần AbsGS (cancellation). [ar5iv+2ECVA+2](https://ar5iv.org/pdf/2404.06109)
* Về mặt SRM, đây là bước làm cho $\rho_{\text{eff}}$ phù hợp forward model, không phải thay story.

Nếu làm đúng, mình kỳ vọng bạn **ít nhất** sẽ kéo về ngang baseline (vì bạn đã thấy gating đưa corr về baseline), và có cửa vượt baseline đặc biệt ở LPIPS/SSIM (đúng pattern của Revising/AbsGS/Pixel-GS). [ar5iv+2ar5iv+2](https://ar5iv.org/pdf/2404.06109)

---

Nếu bạn muốn bước tiếp theo “đinh” để không phí thời gian: mình sẽ viết một checklist triển khai cụ thể theo training loop của 3DGS **cho phiên bản AW-SRM dùng scalar-channel + aux loss** (từng tensor lấy ở đâu, detach thế nào, buffer nào cần thêm, và cách không làm hỏng backward của loss chính). Điều này sẽ biến hướng này thành một kế hoạch code được ngay, không còn là “paper math”.