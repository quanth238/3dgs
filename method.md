\documentclass{article}
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors

% --- Add to preamble (NeurIPS template) ---
\usepackage{amsmath,amssymb}
\usepackage{bm}
\usepackage{algorithm}
\usepackage{algorithmic}

\author{%
  David S.~Hippocampus\thanks{Use footnote for providing further information
    about author (webpage, alternative address)---\emph{not} for acknowledging
    funding agencies.} \\
  Department of Computer Science\\
  Cranberry-Lemon University\\
  Pittsburgh, PA 15213 \\
  \texttt{hippo@cs.cranberry-lemon.edu} \\
  % examples of more authors
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \AND
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
  % \And
  % Coauthor \\
  % Affiliation \\
  % Address \\
  % \texttt{email} \\
}


\begin{document}


\maketitle


\begin{abstract}
  To be written
\end{abstract}

\section{Related Work}

Our work sits at the intersection of explicit 3D Gaussian Splatting (3DGS) optimization, error-driven density control, and measure-theoretic frameworks for inverse problems. Here, we review existing heuristic strategies for densification and the emerging theoretical perspectives that motivate our approach.

\subsection{Adaptive Density Control in 3D Gaussian Splatting}
The original 3DGS framework~\cite{kerbl2023gaussian} introduced Adaptive Density Control (ADC) to dynamically adjust scene capacity. This strategy interleaves standard optimization with heuristic pruning and densification steps. Specifically, Gaussians are cloned or split based on the magnitude of the view-space positional gradient, aggregated over a fixed iteration window~\cite{kerbl20233dgs}. While effective for general reconstruction, this heuristic relies on a proxy metric (gradient magnitude) that often fails to correlate with geometric errors in complex scenarios~\cite{revising2024}.

Several limitations of the naive ADC have been identified. \textbf{Gradient Collision:} \textit{AbsGS}~\cite{ye2024absgs} demonstrated that pixel-wise gradients within a large Gaussian's footprint can cancel each other out due to conflicting directions, resulting in small aggregated gradients despite high reconstruction error. \textbf{View-Averaging Bias:} \textit{Pixel-GS}~\cite{zhang2024pixelgs} noted that averaging gradients across views dilutes the signal for Gaussians visible in many views but under-resolved in specific regions, leading to blur artifacts. To address this, they proposed weighting gradients by the number of covered pixels. \textbf{Structural Mismatch:} \textit{Efficient Density Control (EDC)}~\cite{edc2024} and \textit{ConeGS}~\cite{conegs2025} argued that splitting logic often creates redundant primitives or fails to explore empty space, proposing alternative structural priors or depth-guided spawning.

While these methods patch specific failure modes of ADC, they largely remain heuristic modifications to the selection criterion rather than deriving the densification signal from a unified optimization objective.

\subsection{Visibility-Aware and Error-Driven Refinement}
A parallel line of research shifts the densification trigger from parameter gradients to explicit reconstruction error. \textit{Revising Densification}~\cite{revising2024} proposed distributing pixel-wise error maps (e.g., SSIM or L1 residuals) back to individual Gaussians. Crucially, they identified that error attribution must account for the alpha-blending process; they redistribute error proportional to the Gaussian's contribution (transmittance and opacity) to the final pixel color. This \textit{visibility-aware} attribution is essential for handling occlusion correctly, preventing background primitives from inheriting foreground errors. Similarly, \textit{Pixel-GS}~\cite{zhang2024pixelgs} enforces strict visibility filters to ensure densification only occurs where primitives actively contribute to the image.

Our method aligns with this insight but formalizes it: we treat visibility-weighted error distribution not merely as a heuristic improvement, but as the correct computation of the \textit{adjoint} of the forward rendering operator within a functional gradient framework.

\subsection{Theoretical Frameworks: Measure Optimization and Frank-Wolfe}
Recent theoretical works have sought to ground 3DGS in measure theory. \textit{Splat Regression Models (SRM)}~\cite{srm2024} reformulate 3DGS as a regression problem over a space of mixing measures. They derive the Wasserstein-Fisher-Rao (WFR) gradient flow for splat parameters, showing that the gradient with respect to a splat's amplitude is the integral of the first variation of the objective (the error signal) against the splat's density kernel~\cite{srm2024}.

This formulation connects densification to algorithms for sparse measure optimization, such as the Conditional Gradient method (Frank-Wolfe) and its sliding variants~\cite{denoyelle2019sliding, boyd2017adcg}. In these frameworks, adding a new atom (densification) is equivalent to a Linear Minimization Oracle (LMO) step that selects the atom maximizing correlation with the current residual gradient. While some works have explored sampling perspectives like MCMC for 3DGS~\cite{kheradmand2024mcmc}, they typically rely on stochastic proposals rather than deterministic gradient-guided selection.

Our approach unifies these streams. We adopt the \textit{SRM} perspective that densification is a gradient flow on the measure size (birth process). However, we identify a critical gap: the standard SRM formulation assumes a simplified linear splat model. We reconcile this with the non-linear 3DGS renderer by incorporating the visibility-aware adjoint terms from \textit{Revising Densification}~\cite{revising2024} and the gradient-cancellation awareness of \textit{AbsGS}~\cite{ye2024absgs} (via second moments) into a rigorous Frank-Wolfe selection oracle.

\section{Method}
\label{sec:method}

\subsection{Problem Setup}
We consider the standard 3D Gaussian Splatting (3DGS) setting, where a scene is represented by a set of
anisotropic 3D Gaussians $\mathcal{G}=\{g_i\}_{i=1}^N$.
Each Gaussian $g_i$ has geometry parameters (center and covariance), an opacity parameter, and
appearance parameters (e.g., spherical harmonics coefficients).
Given a calibrated camera $c$, the differentiable rasterizer renders an image
$\hat{\mathbf{I}}_c = \mathcal{R}(\mathcal{G}; c) \in \mathbb{R}^{H\times W \times 3}$.
Training minimizes a photometric objective over a set of training cameras:
\begin{equation}
\min_{\mathcal{G}} \;\; \sum_{c \in \mathcal{C}_{\mathrm{train}}} \ell(\hat{\mathbf{I}}_c, \mathbf{I}^{gt}_c)
\;+\; \lambda\,\mathcal{R}_{\mathrm{reg}}(\mathcal{G}),
\label{eq:train_obj}
\end{equation}
interleaved with \emph{densification} (growing primitives) and pruning.

\vspace{0.2em}
\noindent\textbf{Goal.}
We aim to replace heuristic densification criteria with a \emph{renderer-consistent} score motivated by
Splat Regression Models (SRM), while remaining faithful to the practical realities of 3DGS
(alpha compositing, occlusion, and limited densification budgets).


\subsection{SRM View: Densification as a Restricted Conditional-Gradient Step}
SRM models a function $f_\mu$ as a mixture (measure) of parameterized kernels (``splats''):
\begin{equation}
f_\mu(x) = \int v\,\rho_{A,b}(x)\; \mu(dv,dA,db).
\label{eq:srm_model}
\end{equation}
For a functional $F(f)$, SRM defines the first variation $\delta F[f](x)$ and shows that the Wasserstein
gradient component w.r.t.\ the output vector $v$ takes the form
\begin{equation}
\nabla_v F(f_\mu)(v,A,b) \;=\; \int \delta F[f_\mu](x)\,\rho_{A,b}(x)\,\pi(dx),
\label{eq:srm_grad_v}
\end{equation}
i.e., a correlation between an error-like signal $\delta F$ and the splat density.

In sparse inverse problems, conditional-gradient methods (Frank--Wolfe / ADCG) add new atoms by
maximizing such correlations, then refine existing atoms via local descent.
In 3DGS, we adopt the same \emph{principle} but use a restricted atom family: we only create new
Gaussians via the standard 3DGS operations (clone and split) so that densification remains stable and
GPU-friendly.

\subsection{Why Footprint-Only Scores Fail Under Alpha Compositing}
A key subtlety is that $\delta F$ in \eqref{eq:srm_grad_v} is defined in the space of the model output $f$.
In novel view synthesis, the loss is applied to rendered pixels, and the renderer $\mathcal{R}$ is the
measurement operator. Thus, the signal that should be ``correlated with a kernel'' is not a raw image-space
gradient integrated over geometric overlap, but rather the pixel signal \emph{pulled back through the
adjoint of the renderer}.
For 3DGS, this adjoint contains \emph{visibility/transmittance} factors arising from front-to-back
alpha compositing.

This observation matches recent densification improvements: per-pixel errors must be attributed to
Gaussians proportionally to their true contribution under compositing, rather than by footprint overlap.

\subsection{Adjoint-Consistent Error Attribution via Alpha-Weighted Moments}
Let $\alpha_{i,c}(u)$ denote the \emph{alpha-compositing coefficient} of Gaussian $i$ at pixel $u$
when rendering camera $c$, i.e., the scalar weight by which Gaussian $i$ contributes to the pixel under
front-to-back compositing (including transmittance/visibility).
We build densification statistics using an auxiliary, nonnegative per-pixel signal:
\begin{equation}
E_c(u) \;\ge\; 0,
\label{eq:error_map}
\end{equation}
chosen as an error map (e.g., per-pixel $\ell_1$ residual, or a SSIM-derived dissimilarity).
Crucially, we \emph{detach} $E_c$ from the training graph so it is used only for densification decisions.

For each Gaussian, we accumulate three alpha-weighted moments over a collection window of cameras
(or mini-batches) $\mathcal{B}$:
\begin{align}
Z_i &= \sum_{c\in\mathcal{B}} \sum_{u} \alpha_{i,c}(u), \label{eq:moment_Z}\\
M_i &= \sum_{c\in\mathcal{B}} \sum_{u} E_c(u)\,\alpha_{i,c}(u), \label{eq:moment_M}\\
Q_i &= \sum_{c\in\mathcal{B}} \sum_{u} E_c(u)^2\,\alpha_{i,c}(u). \label{eq:moment_Q}
\end{align}
$M_i$ is the \emph{attributed error mass} of Gaussian $i$ (how much error it is responsible for, under the
renderer), while $Q_i$ captures the \emph{energy} of error attributed to that Gaussian.

We further define the normalized mean and a second-central-moment statistic:
\begin{equation}
\mu_i = \frac{M_i}{Z_i+\varepsilon}, \qquad
\mathrm{Var}_i = \frac{Q_i}{Z_i+\varepsilon} - \mu_i^2,
\label{eq:mean_var}
\end{equation}
where $\varepsilon$ is a small constant.

\subsection{Clone vs.\ Split Scores}
Clone and split serve different purposes in 3DGS: cloning increases local capacity in under-reconstructed
regions, while splitting targets overly large primitives that smear high-frequency structure.
We therefore use two complementary scores:

\vspace{0.2em}
\noindent\textbf{Clone score (under-reconstruction).}
We prioritize Gaussians that carry large attributed error mass:
\begin{equation}
s_i^{\mathrm{clone}} = M_i.
\label{eq:clone_score}
\end{equation}
This directly increases capacity where the renderer attributes persistent error.

\vspace{0.2em}
\noindent\textbf{Split score (high-frequency / cancellation).}
A large $\mathrm{Var}_i$ indicates that the pixel signal attributed to $i$ varies strongly within its footprint:
a single primitive cannot explain the local structure with a coherent correction, which is typical of
fine texture or edges.
We combine this with a monotone size factor $\psi(r_i)$ (e.g., $\psi(r_i)=r_i$ where $r_i$ is the
screen-space radius):
\begin{equation}
s_i^{\mathrm{split}} =
\Big(Q_i - \frac{M_i^2}{Z_i+\varepsilon}\Big)\,\psi(r_i).
\label{eq:split_score}
\end{equation}
This score is robust to sign cancellation (it depends on energy), and naturally targets large primitives in
detail-rich regions.

\subsection{Budgeted Restricted Oracle and Densification Procedure}
At a densification step, we select two disjoint sets:
(i) top-$K_{\mathrm{clone}}$ Gaussians by $s_i^{\mathrm{clone}}$ among ``small'' primitives,
(ii) top-$K_{\mathrm{split}}$ Gaussians by $s_i^{\mathrm{split}}$ among ``large'' primitives.
We then apply the standard 3DGS clone/split operations to these sets, while enforcing a per-step primitive
budget and a global maximum primitive count.

\begin{algorithm}[t]
\caption{Alpha-Weighted Moment Densification (AWMD)}
\label{alg:awmd}
\begin{algorithmic}[1]
\STATE \textbf{Input:} current Gaussians $\mathcal{G}$, cameras $\mathcal{B}$, budgets $K_{\mathrm{clone}},K_{\mathrm{split}}$
\STATE Compute detached per-pixel maps $\{E_c\}_{c\in\mathcal{B}}$
\STATE Accumulate $(Z_i,M_i,Q_i)$ via \eqref{eq:moment_Z}--\eqref{eq:moment_Q}
\STATE Compute scores $s_i^{\mathrm{clone}}, s_i^{\mathrm{split}}$ via \eqref{eq:clone_score}, \eqref{eq:split_score}
\STATE Select sets $\mathcal{S}_{\mathrm{clone}}, \mathcal{S}_{\mathrm{split}}$ under size constraints and budgets
\STATE Apply clone on $\mathcal{S}_{\mathrm{clone}}$; apply split on $\mathcal{S}_{\mathrm{split}}$
\STATE Apply pruning / opacity reset schedule as in 3DGS, with clone-opacity correction if used
\end{algorithmic}
\end{algorithm}

\subsection{Implementation: Computing Alpha-Weighted Moments}
Directly computing $\alpha_{i,c}(u)$ for all pixels is expensive, but 3DGS rasterization already evaluates
these coefficients during rendering. We provide two practical implementations:

\vspace{0.2em}
\noindent\textbf{(A) Exact attribution via auxiliary scalar rendering.}
We assign each Gaussian a set of auxiliary scalar attributes and render them through the same alpha
compositing as color. We then form an auxiliary objective as a dot product between rendered scalars and
detached pixel weights, and obtain $(Z_i,M_i,Q_i)$ as gradients w.r.t.\ the auxiliary scalars.
All Gaussian geometry/appearance parameters are detached in this pass so the auxiliary objective does not
affect training.

\vspace{0.2em}
\noindent\textbf{(B) Fast approximation via rasterizer-aware CUDA reduction.}
We reuse the rasterizer's tile lists and depth ordering to approximate transmittance within each tile,
accumulating per-Gaussian contributions using a small set of sample points per tile. This preserves the
core visibility mechanism of alpha compositing while remaining lightweight enough to run at the standard
densification interval.



\bibliographystyle{plainnat}
\bibliography{references}

\end{document}