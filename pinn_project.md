> From: https://chatgpt.com/c/69b24f32-74e4-832e-9d8e-2767ce4d651b

# you asked

message time: 2026-03-12 01:29:42

which one do you think I would enjoy the most? what about learn the most? what about useful for my future jobs?

---

# chatgpt response

I’m comparing the three proposals against your past preferences: mathematically rich research, Julia/SciML, HPC, and career signaling. The answer is not “which is objectively best,” but “best for Angelo.”

Based on what you usually gravitate toward, I think:

**Enjoy the most:** **Physics-Constrained Flow Matching**  
**Learn the most overall:** **Physics-Constrained Flow Matching**  
**Most useful for future jobs:** **TensorCore abstractions** for systems/HPC roles, but **PCFM** for ML research / SciML / PhD-style paths.

My honest “for Angelo” ranking would be:

**1) PCFM**  
**2) TensorCore**  
**3) Geo**

Why:

### 1) What you’d probably enjoy the most: **PCFM**
This one feels the most “you.” It mixes nonlinear optimization, AD, GPU computation, solver design, and scientific ML, while still being mathematically meaningful. The proposal is about extending PCFM.jl from linear constraints to nonlinear ones, designing a modular projection-solver interface, and exploring backends like Optimization.jl, MadNLP+ExaModels, and Reactant.jl. It also has clear evaluation metrics and explicit publication/open-source potential. 

That is a very nice sweet spot for you: not just “build stuff,” not just “theory,” and not just “systems for systems’ sake.” It has enough structure to feel researchy, and enough implementation depth to stay fun.

### 2) What you’d likely learn the most: **PCFM**, with one caveat
If by “learn the most” you mean **broadest intellectual growth**, I’d still say **PCFM**. It touches constrained optimization, Jacobians/Newton-style methods, block structure, GPU acceleration, XLA/Reactant, and solver abstraction design. The proposal itself explicitly ties the work to adjoints, AD, GPU parallelism, and nonlinear PDE-style constraints. 

The caveat is that **TensorCore** may teach you the most **low-level systems/HPC-specific material**. That project is about creating a portable abstraction for TensorCore/MatrixCore-style operations in Julia, integrated with KernelAbstractions, with backend-specific mappings to NVIDIA/AMD-style matrix hardware. That is deeper exposure to hardware-aware performance engineering, mixed-precision kernels, and portability abstractions. 

So:
- **broadest learning:** PCFM
- **deepest low-level systems learning:** TensorCore

### 3) What is most useful for future jobs?
This depends a lot on which jobs you mean.

If you mean **ML research, scientific ML, research engineering, PhD applications, DeepMind/OpenAI/Google Research-type stories**, then **PCFM** is the strongest fit. It is close to modern generative modeling, constrained inference, SciML, optimization, and open-source numerical software, and the proposal explicitly mentions a possible publication path. 

If you mean **GPU systems, ML infrastructure, compilers, performance engineering, CUDA-like kernel work, hardware-software co-design**, then **TensorCore** is probably the most job-useful. A portable abstraction for specialized matrix engines is exactly the kind of thing that signals strong systems taste and rare infrastructure skills. 

If you mean **climate tech / agentic AI / end-to-end applied package building**, then **Geo** has value. It combines simulation, optimization, LLM-agent orchestration, economic modeling, reporting, package release, and a paper/report deliverable.   
But for you personally, it feels less naturally aligned with your strongest motivations than the other two.

### My bottom line
If you want **one recommendation**, I’d choose **PCFM**.

It seems like the best combination of:
- enjoyment,
- mathematical depth,
- relevance to your current interests,
- good research upside,
- and still-solid career value.

I’d choose **TensorCore instead** only if you want to deliberately push yourself toward a more **systems/HPC/compiler/performance** identity.

And I’d choose **Geo** only if what excites you most is building an end-to-end project with **agentic AI + applied simulation + climate/energy relevance** rather than a tighter technical core.

My one-line summary:

**PCFM = best fit for you. TensorCore = best strategic detour. Geo = coolest applied wildcard.**

If you want, I can also give you a **decision matrix with categories like fun / depth / publishability / workload / resume value / risk**.

---

# you asked

message time: 2026-03-12 01:36:02

ok, I'll try to see both then... basically this is for a class project and they said if someone was interested they should reach out by email

what should i say? should i ask about a tutorial on it / introductionary in-person meeting / easy task to get familiarity?

---

# chatgpt response

Yes — I would **reach out now**, and I’d frame it as:

- you’re genuinely interested,
- you’d appreciate a brief conversation to understand the project better,
- and you’d be happy to start with a small scoped task or reading/tutorial to get up to speed.

I would **not** say “easy task,” because it can sound like you want something trivial.  
Better wording is: **“a small starter task,” “an initial reading/task,”** or **“a good way to ramp up.”**

A strong email usually asks for two things at most:
1. a short meeting or chat, and  
2. guidance on how to get familiar with the project.

You can send something like this:

**Email Title: Interest in class project opportunity**

Hi [Professor/Name],

I hope you are doing well. I’m writing because I’m very interested in the project you mentioned in class and would love to learn more about it.

The topic seems closely aligned with my interests, and I would be excited to explore whether I could contribute. If you are available, I would really appreciate the chance to meet briefly sometime soon for an introductory conversation about the project and what working on it might involve.

Also, if there is a good way to start getting familiar with the project, I’d be very happy to begin with any suggested background reading, tutorial material, or a small initial task.

Thank you very much for your time.

Best,  
Angelo Farfan

A slightly warmer version, if you want it to sound more natural and less formal:

**Email Title: Interested in learning more about the class project**

Hi [Professor/Name],

I wanted to reach out because I’m very interested in the project you mentioned in class.

It sounds like a topic I’d really enjoy working on, and I’d love to learn more about it. If you have time, I’d really appreciate the chance to meet briefly sometime soon for an introductory chat.

I’d also be happy to start by reading background material or working through a small starter task if that would be helpful for getting familiar with the project.

Thank you, and I look forward to hearing from you.

Best,  
Angelo Farfan

My recommendation: **ask for the meeting and mention openness to reading/tutorial/starter task in the same email.** That makes you sound proactive without sounding needy.

If you want, I can tailor it to one of the two projects specifically so it sounds more informed.

---

# you asked

message time: 2026-03-12 01:38:24

oh wait... it seems I forgot to attach one more description... there you go

---

# chatgpt response

The new MOR proposal does change the picture a bit. It looks more aligned with your math-symbolics-Julia side than Geo, and possibly closer to your natural taste than TensorCore unless you want a systems-heavy detour.

Ah — this **does** change my view a bit.

With **Intrusive MOR via Symbolic Computation in SciML**, my updated “for Angelo” ranking would probably be:

**1) Physics-Constrained Flow Matching**  
**2) Intrusive MOR**  
**3) TensorCore abstractions**  
**4) Geo**

Why I’d move **Intrusive MOR** up:
- it is very Julia/SciML-native,
- it has real mathematical content,
- it involves symbolic computation and equation transformations,
- and it ends in a concrete implementation/PR to `ModelOrderReduction.jl`, which is nice for both research and resume value. The proposal explicitly centers on automating intrusive MOR methods like Lift-and-Learn or PCE using `Symbolics.jl`, `ModelingToolkit.jl`, and integration into `ModelOrderReduction.jl`. 

So in terms of your earlier categories, I’d now say:

- **Enjoy the most:** still probably **PCFM**, but **Intrusive MOR** is now a very real contender.
- **Learn the most:** **PCFM** or **Intrusive MOR**, depending on whether you want more optimization/solver/GPU exposure or more symbolic/model-reduction exposure.
- **Most useful for future jobs:**
  - **TensorCore** for low-level HPC/systems/performance roles,
  - **PCFM** for SciML / ML research engineering / numerical ML,
  - **Intrusive MOR** for a more researchy Julia scientific-computing profile.

For your email: yes, I think you should ask for
- a brief introductory conversation,
- and a good way to ramp up, such as background reading or a small starter task.

I would **not** ask for a “tutorial” too directly, because that can sound passive. Better:
- “a brief meeting to learn more,”
- “recommended background reading,”
- “a small starter task,”
- “a good way to get familiar with the project.”

Since you want to explore **both**, I’d send an email like this:

**Email Title: Interest in project opportunities**

Hi [Professor/Name],

I hope you are doing well. I wanted to reach out because I am very interested in the project opportunities that were mentioned in class, especially [Project A] and [Project B].

Both topics seem closely related to areas I would enjoy learning more about, and I would love to get a better sense of what working on either project would involve. If you are available, I would really appreciate the chance to meet briefly sometime soon for an introductory conversation.

Also, if there is a good way to start getting familiar with the projects, I would be very happy to begin with any recommended background reading or a small starter task.

Thank you very much for your time.

Best,  
Angelo Farfan

If you want it to sound a bit more informed, you can make it slightly more specific. For example, if the two are **PCFM** and **Intrusive MOR**:

**Email Title: Interest in class project opportunities**

Hi [Professor/Name],

I hope you are doing well. I wanted to reach out because I am very interested in the class project opportunities you mentioned, particularly the Physics-Constrained Flow Matching project and the Intrusive Model Order Reduction project.

Both seem very appealing to me, especially because I am interested in scientific machine learning, mathematical modeling, and computational methods in Julia. I would love to learn more about what each project would involve and where I could be most useful.

If you have time, I would really appreciate the chance to meet briefly for an introductory conversation. I would also be very happy to start with any recommended background reading or a small starter task to get familiar with the project.

Thank you very much for your time.

Best,  
Angelo Farfan

My recommendation: **ask for a brief meeting + background reading/small starter task in the same email.** That comes across as proactive and serious.

I can also make this sound more casual, more professor-like, or tailor it to the exact person you’re emailing.

---

# you asked

message time: 2026-03-12 01:57:24

Ok, let's lock in for the Physics-Constrained Flow Matching one... I need to email Utkarsh... What do I write? Also note that I would highly prefer to either meet this Friday (at any time) or virtually on the weekend or next week again only on Friday (at any time) or Thursday night

---

# chatgpt response

Here’s a good version to send:

**Email Title: Interest in the Physics-Constrained Flow Matching project**

Hi Utkarsh,

I hope you are doing well. I wanted to reach out because I am very interested in the Physics-Constrained Flow Matching project you shared for the class.

The project sounds very exciting to me, especially because of its connection to scientific machine learning, constrained optimization, and Julia/SciML tools. I would love to learn more about the project and what working on it would involve.

If you are available, I would really appreciate the chance to meet briefly for an introductory conversation. My strongest preference would be to meet in person this Friday at any time. If that is not convenient, I would also be happy to meet virtually over the weekend, or next week either on Thursday night or Friday at any time.

Also, if there is any background reading or a small starter task you would recommend so I can begin getting familiar with the project, I would be very happy to do that.

Thank you very much for your time, and I look forward to hearing from you.

Best,  
Angelo Farfan

A slightly warmer, more natural variant:

**Email Title: Interested in the Physics-Constrained Flow Matching project**

Hi Utkarsh,

I hope you are doing well. I wanted to reach out because I am very interested in the Physics-Constrained Flow Matching project.

It seems very closely aligned with the kind of work I would enjoy, and I would love to learn more about it. If you have time, I would really appreciate the chance to meet briefly and hear more about the project and how I could get up to speed.

My ideal preference would be to meet in person this Friday at any time. Otherwise, I would also be happy to meet virtually over the weekend, or next week on Thursday night or Friday at any time.

If there is any reading, tutorial material, or small starter task you would recommend beforehand, I would also be glad to take a look.

Thank you, and I look forward to hearing from you.

Best,  
Angelo Farfan

I’d send the first one. It sounds proactive and clear without overdoing it.

---

# you asked

message time: 2026-03-12 02:06:11

how about this?

Hi Utkarsh,

I am reaching out since I liked the project description for Physics-Constrained Flow Matching.

It seems aligned with the kind of work I would enjoy. I would like to learn more about the project and what a suitable timeline would look like within the team.

If you have time, we could meet briefly for an introductory conversation or how I could get up to speed. My strongest preference would be to meet in person this Friday 13 between 3-5pm (just 30min I guess). We could also meet virtually on this weekend.

Also, is there any background reading (besides the repos) or a smaller starter task you would recommend?

Look forward to hearing from you.

Best,
Angelo Farfan

---

# chatgpt response

Yes — this is already pretty solid. I’d just smooth out a few phrases so it sounds a bit more natural and polished.

Main tweaks I’d make:
- “since I liked the project description” → “because I was very interested in the project description”
- “or how I could get up to speed” → “to learn more about the project and how I could get up to speed”
- “Friday 13 between 3–5pm” → “Friday, March 13, between 3–5pm”
- “Look forward to hearing from you” → “I look forward to hearing from you”

Here is a cleaner version:

**Email Title: Interest in the Physics-Constrained Flow Matching project**

Hi Utkarsh,

I am reaching out because I was very interested in the project description for Physics-Constrained Flow Matching.

It seems closely aligned with the kind of work I would enjoy, and I would love to learn more about the project and what a suitable timeline would look like within the team.

If you have time, I would really appreciate the chance to meet briefly for an introductory conversation to learn more about the project and how I could get up to speed. My strongest preference would be to meet in person this Friday, March 13, between 3–5pm, even just for around 30 minutes. Otherwise, I would also be happy to meet virtually over the weekend.

Also, is there any background reading, besides the repositories, or a small starter task you would recommend?

I look forward to hearing from you.

Best,  
Angelo Farfan

A tiny stylistic note: “within the team” is fine, but “for the project” also works and sounds slightly more natural here.

---

# you asked

message time: 2026-03-12 02:11:08

quick question: You are aware that Utkarsh knows nothing about me, right? I mean, he could sort of infere that I'm enrolled in Edelman's class but nothing else.

---

# chatgpt response

Yes — in that case you should add **one brief sentence of context** about who you are.

Right now the draft assumes a bit too much familiarity. Since Utkarsh likely only knows you as someone from Edelman’s class, it would be better to identify yourself early, for example:

- that you’re a student in Edelman’s class
- that you’re reaching out about the PCFM class project
- optionally, one short phrase about your interests or background

You do **not** need a long self-introduction. Just enough so the email does not feel contextless.

A better opening would be something like:

“Hi Utkarsh,  
I’m Angelo Farfan, a student in Prof. Edelman’s class. I’m reaching out because I was very interested in the project description for Physics-Constrained Flow Matching.”

Or slightly stronger:

“Hi Utkarsh,  
I’m Angelo Farfan, a student in Prof. Edelman’s class, and I’m reaching out because I was very interested in the Physics-Constrained Flow Matching project description.”

Then later you can keep the rest mostly the same.

I would also slightly adjust the “timeline within the team” line, because that can sound a bit formal or as if you are already joining. Something like:

“I would love to learn more about the project, what working on it would involve, and what a suitable timeline might look like.”

So yes: **add a short self-ID sentence.** That is the main thing missing.

A polished version would be:

**Email Title: Interest in the Physics-Constrained Flow Matching project**

Hi Utkarsh,

I’m Angelo Farfan, a student in Prof. Edelman’s class, and I’m reaching out because I was very interested in the project description for Physics-Constrained Flow Matching.

It seems closely aligned with the kind of work I would enjoy, and I would love to learn more about the project, what working on it would involve, and what a suitable timeline might look like.

If you have time, I would really appreciate the chance to meet briefly for an introductory conversation and to learn how I could get up to speed. My strongest preference would be to meet in person this Friday, March 13, between 3–5pm, even just for around 30 minutes. Otherwise, I would also be happy to meet virtually over the weekend.

Also, is there any background reading, besides the repositories, or a small starter task you would recommend?

I look forward to hearing from you.

Best,  
Angelo Farfan

This version fixes the “who are you?” issue nicely.

---

# you asked

message time: 2026-03-19 10:23:10

Do you have any ideas? i.e. actually propose a project that is not the ones I shared

---

# chatgpt response

I have a few that feel much more “you” than generic class-project ideas: sparse/differentiable numerical methods, structure preservation, and Julia-native tooling with a clean experiment story.

Yes — and honestly, I think this could be a better route for you than just inheriting one of the listed ones.

If I were proposing a project **for you**, I would bias toward something that has:

- real mathematical structure,
- Julia / SciML relevance,
- a clean implementation target,
- and an experiment story that is realistic for a class project.

Here are the ideas I think fit you best.

## My favorite one for you

### 1) Differentiable nonlinear projection layers for scientific ML
**Core idea:** build a reusable Julia module for projecting onto **nonlinear constraint sets** inside ML pipelines, with gradients computed either by unrolling or implicit differentiation.

This is related in spirit to the Physics-Constrained Flow Matching idea, but it is broader and more “tool-building.” Instead of focusing only on one flow-matching application, the project becomes:

> given a constraint `c(x)=0` or `c(x) <= 0`, design a projection / correction layer  
> `x -> Π_C(x)`  
> that is numerically stable, modular, and differentiable.

### Why this is good
It hits a lot of your strengths:
- constrained optimization,
- Jacobians / linear solves,
- AD,
- numerical methods,
- modular software design.

### Possible scope
- Start with equality constraints.
- Implement Newton / Gauss-Newton / SQP-style projection.
- Compare:
  - unrolled differentiation,
  - implicit differentiation,
  - maybe approximate Jacobians.
- Benchmark on simple physical systems or toy manifolds.

### Deliverable
A Julia package extension or clean repo with:
- a generic projection interface,
- a few nonlinear examples,
- timing + gradient-quality experiments.

This feels very plausible, technically interesting, and strong for a class project.

---

## Other strong ideas

### 2) Mixed-precision and iterative refinement for differentiable scientific computing
**Core idea:** study whether you can use **mixed precision** for the linear algebra inside optimization / implicit layers / ODE adjoints while preserving gradient accuracy.

This is nice because it combines:
- numerical linear algebra,
- hardware-awareness,
- optimization,
- practical performance.

### Example question
When solving the linear systems that arise in Newton steps or implicit differentiation, can we do most of the work in lower precision and recover accuracy with iterative refinement?

### Why it’s strong
This gives you:
- a very concrete benchmark story,
- relevance to future jobs,
- real math,
- and not-too-large implementation scope.

### Nice framing
“Fast and stable mixed-precision primitives for differentiable optimization layers in Julia.”

This one is a bit more systems/numerics than pure ML, which I think suits you well.

---

### 3) Structure-preserving neural ODE / PDE training
**Core idea:** compare standard training against training with explicit preservation of structure:
- conservation laws,
- symmetries,
- invariants,
- positivity,
- divergence constraints.

Instead of “just fit the data,” the question is:
> can we impose physically meaningful structure in a way that improves generalization or stability?

### Variants
- Hamiltonian-like systems: preserve energy approximately.
- Transport / density systems: preserve mass or positivity.
- Incompressible flow toy models: preserve divergence-free structure.

### Why this is good
It is mathematically interpretable and experimentally neat. It also has a nice class-project narrative:
1. identify a structure,
2. enforce it in training or architecture,
3. compare behavior.

This is less tooling-heavy and more “scientific question.”

---

### 4) Intrusive vs non-intrusive model reduction in SciML
**Core idea:** instead of only implementing intrusive MOR, do a **comparative project**:
- intrusive MOR,
- data-driven / operator-inference style reduction,
- maybe autoencoder-based latent reduction.

Then ask:
> when does symbolic access to the equations really buy you something?

### Why this is good
You’d learn a lot, and the project has a sharp conceptual punchline.

### Great outcome
A small benchmark suite on a few dynamical systems showing tradeoffs in:
- accuracy,
- speed,
- interpretability,
- implementation complexity.

This is more “research-comparison” than “build one new method,” which can be very smart for a class project.

---

### 5) Automatic sparsity-exploiting implicit differentiation in Julia
**Core idea:** many optimization and scientific ML problems have sparse Jacobians / KKT systems, but autodiff pipelines often don’t exploit that well.

Project goal:
> build a prototype that automatically detects sparsity structure and uses sparse linear algebra in implicit differentiation.

### Why I think you’d like it
This is very close to your taste:
- sparse matrices,
- structure,
- fast solvers,
- mathematically meaningful implementation.

### Deliverable
A prototype wrapper around a small family of problems:
- constrained layers,
- implicit equations,
- steady-state solves,
- optimization layers.

This is probably one of the most “Angelo-coded” ideas here.

---

### 6) Adaptive solver selection for optimization / SciML pipelines
**Core idea:** create a lightweight meta-layer that chooses among solvers or linear algebra backends depending on problem structure.

For example:
- dense vs sparse,
- small vs medium,
- well-conditioned vs ill-conditioned,
- linear vs nonlinear constraints.

### Why it’s nice
It’s practical, modular, and class-project friendly.

### Why it may be less ideal for you
It can become too engineering-heavy unless you anchor it with a real mathematical question.

---

## What I would actually recommend

If you want the project that feels **most natural for you**, I’d rank them:

1. **Automatic sparsity-exploiting implicit differentiation**
2. **Differentiable nonlinear projection layers**
3. **Mixed-precision + iterative refinement for differentiable SciML**
4. **Intrusive vs non-intrusive MOR comparison**
5. **Structure-preserving neural ODE/PDE training**
6. **Adaptive solver selection**

## The one I’d pitch first
I think the strongest pitch is:

**“Differentiable sparse nonlinear projection layers for scientific machine learning in Julia.”**

Why this one:
- close enough to the PCFM theme that it sounds relevant,
- but distinct enough to be your own project,
- mathematically rich,
- implementation-friendly,
- and not too broad.

A clean project statement would be:

> Build a modular Julia framework for nonlinear projection/correction layers that enforce physical constraints in machine learning models, with a focus on sparse Jacobians, stable numerical solvers, and differentiable implementations via implicit differentiation.

That sounds serious, original, and very on-brand for you.

## If you want a more researchy angle
Then I’d instead pitch:

**“When does symbolic structure help model reduction?”**

That becomes:
- intrusive MOR,
- operator inference / data-driven MOR,
- comparison on benchmark systems.

Very elegant, very class-project appropriate, and easy to explain.

## If you want stronger job value
Then I’d pitch:

**“Mixed-precision sparse linear algebra for differentiable scientific computing.”**

That has clearer systems/HPC signaling.

If you want, I can turn my top 3 into **actual 1-paragraph project proposals you could email to a professor or TA**.

---

# you asked

message time: 2026-03-19 10:26:20

I like 1,2,3,5 and low-key 3 and 5 sound interesting... Could you give more details? Would any of them help/produce a PINN? Are they actually aligned with the course??

---

# chatgpt response

I’m checking the uploaded project descriptions so I can answer the “course alignment” part concretely rather than just guessing from memory.

I’ve confirmed that the official project descriptions explicitly tie PCFM to adjoints, AD, GPU parallelism, XLA/Reactant, and sparse NLP on GPU, so I can map your custom ideas against the actual course themes now.

Yes — all four can be made **course-aligned**, but they differ a lot in **how naturally** they fit 18.337.

Given the official proposals, the course clearly likes projects around:
- adjoints / automatic differentiation,
- Julia + SciML tooling,
- GPU-aware implementation,
- sparse / structured numerical methods,
- and open-source research software with benchmarks. That is explicit in the PCFM proposal, which ties the work directly to adjoints, AD, GPU parallelism, Reactant/XLA, and sparse structured NLP; and in the MOR proposal, which emphasizes Symbolics/ModelingToolkit, algorithmic implementation, and even custom student-proposed methods.  

So the answer is: **yes, these can absolutely fit the course** if pitched the right way.

## My quick take

If you want the idea **closest to a PINN**, choose **3**.  
If you want the idea **most aligned with your taste and with the course’s computational themes**, choose **5**.  
If you want something **closest to Utkarsh’s PCFM project but still your own**, choose **1**.

---

## 1) Differentiable nonlinear projection layers for scientific ML

### What it really is
You build a reusable module that takes a tentative state `z` and projects/corrects it onto a nonlinear constraint set:
- energy conservation,
- mass conservation,
- boundary constraints,
- positivity-like constraints,
- manifold constraints.

So instead of only training a model to “hopefully” satisfy physics, you add a correction layer:
- `z -> Π_C(z)` or
- `z -> corrected(z)`

computed by Newton / Gauss-Newton / SQP / interior-point style steps.

### Why it is aligned with the course
This is very close in spirit to the official PCFM project: the proposal literally centers on differentiating constraints, forming Newton steps, exploiting block/sparse Jacobian structure, and making the implementation composable in Julia. 

### Could it help produce a PINN?
**Yes, but indirectly.**
This is **not itself a PINN**, but it can become:
- a projection layer inside a PINN,
- a post-processing correction step for PINN outputs,
- or a hard-constraint alternative to soft penalty terms.

That is actually pretty appealing: standard PINNs often use soft penalties, while your project would explore **hard or semi-hard constraint enforcement**.

### Good version of the project
A good class-project framing would be:

> Implement differentiable nonlinear projection layers in Julia for physics-constrained ML, compare unrolled differentiation vs implicit differentiation, and benchmark on one or two PDE/ODE toy problems.

That would be very solid.

### Risk
The danger is scope creep. You need to keep it to:
- one clean interface,
- one or two solvers,
- one or two benchmark problems.

---

## 2) Mixed-precision and iterative refinement for differentiable scientific computing

### What it really is
You study the linear algebra inside:
- Newton steps,
- implicit differentiation,
- adjoint solves,
- maybe stiff solver internals,

and ask whether lower precision can give speedups **without ruining gradients**.

### Why it is aligned
It matches the course if you frame it around:
- differentiable solvers,
- GPU execution,
- accuracy/performance tradeoffs,
- Julia implementation.

This is not as directly mirrored by the uploaded proposals as 1 or 5, but it still fits the same ecosystem and computational spirit. The PCFM proposal is already concerned with backend design, GPU acceleration, and solver performance, so a mixed-precision numerical-methods project is definitely in-family. 

### Could it help produce a PINN?
**Only indirectly.**
It would not “produce a PINN” in the conceptual sense. But if your PINN training or implicit residual solve is expensive, this could make it faster.

So:
- **PINN-enabling?** yes
- **PINN-centric?** no

### Good version of the project
A clean version would be:

> Evaluate mixed-precision Newton/KKT/adjoint linear solves for differentiable scientific ML workloads in Julia, using iterative refinement to recover accuracy.

### Risk
This one can become too “benchmarky” and less conceptually sharp unless you pick a specific target:
- differentiable optimization layers,
- constrained projection,
- or adjoint-based training.

I would not choose this one unless you specifically want a numerics/performance project.

---

## 3) Structure-preserving neural ODE/PDE training

This is the one you low-key liked, and I think that makes sense.

### What it really is
You train a neural ODE / PDE-related model while enforcing or encouraging structure such as:
- energy conservation,
- mass conservation,
- symplecticity / Hamiltonian behavior,
- positivity,
- divergence-free constraints,
- boundary-condition consistency.

Then you compare against a baseline that ignores structure.

### Could it produce a PINN?
**Yes — this is the closest one to a PINN.**

You could do this in at least three ways:

**Version A: true PINN**
- Use a neural network `u_theta(x,t)`,
- penalize PDE residuals and boundary conditions,
- add a structure-preserving mechanism beyond the usual soft PINN penalties.

**Version B: neural ODE / latent dynamics**
- Learn dynamics from trajectories,
- impose conservation laws or invariants during training.

**Version C: hybrid**
- Use a PINN-like residual loss plus a hard/soft structure-preserving layer.

So yes, **3 can absolutely be made into a PINN project**.

### Why it is aligned
It is aligned **if it stays computational and method-oriented**, not just “train one model and show plots.”  
The official projects emphasize scientific ML, adjoints/AD, algorithmic implementation, and physics structure. The MOR proposal even cites “Lift and Learn: Physics-informed machine learning for large-scale nonlinear dynamical systems,” so physics-informed learning is very much in the orbit of accepted project themes. 

### Best framing
I would frame it as:

> Hard vs soft structure enforcement in neural ODE/PINN training in Julia: a comparison of penalty methods, projection methods, and architecture-level constraints.

That is actually a very nice class project.

### Why I think this one is interesting
It gives you:
- a clear scientific question,
- a natural experiment,
- and a direct connection to PINNs.

### Risk
You need to avoid making it too broad. Pick **one structure** and **one benchmark family**.  
For example:
- Burgers / diffusion-like PDE with mass conservation,
- or a Hamiltonian ODE with energy preservation.

---

## 5) Automatic sparsity-exploiting implicit differentiation in Julia

This is the one I think is **most naturally strong** for you.

### What it really is
Many scientific ML pipelines need gradients through equations of the form:
- `F(z, θ) = 0`,
- optimization problems,
- projection layers,
- steady states,
- constrained solves.

Implicit differentiation says you can get gradients by solving linear systems involving Jacobians/KKT matrices instead of backpropagating through every iteration.

Your project would be:
- detect or expose sparsity structure,
- use sparse linear algebra automatically,
- and benchmark against dense / naive AD.

### Why it is aligned
This is almost perfectly aligned with the PCFM proposal’s logic:
- nonlinear constraints,
- Jacobians,
- Newton steps,
- block-diagonal structure,
- backend design,
- sparse exploitation,
- GPU-aware numerical solving. 

It also meshes with the MOR proposal’s Julia/SciML/tooling emphasis: building algorithmic infrastructure rather than only running an application. 

### Could it help produce a PINN?
**Yes, but as infrastructure rather than as the front-facing model.**

This would be useful if your PINN or physics-informed method includes:
- implicit layers,
- PDE-constrained optimization,
- projection steps,
- equilibrium models,
- differentiable solvers.

So it can be a **backend for advanced PINN-like methods**, but not itself a PINN.

### Best framing
I would pitch it like this:

> Build a Julia prototype for sparse implicit differentiation in scientific ML, automatically exploiting Jacobian/KKT sparsity for constrained layers and implicit solves.

That is very 18.337-coded.

### Why I like it for you
It has:
- real math,
- real performance questions,
- a software artifact,
- and a strong “this person understands structure” signal.

### Risk
You need one clean benchmark family. For example:
- nonlinear projection layers,
- or constrained equilibrium problems,
- not five different applications.

---

## Which ones can genuinely be a PINN?

From most to least:

**3 > 1 > 5 > 2**

- **3**: yes, naturally a PINN or PINN-adjacent project.
- **1**: yes, can be integrated into a PINN as hard-constraint enforcement.
- **5**: useful backend for more sophisticated PINN/implicit PDE training.
- **2**: mostly performance infrastructure, not a PINN idea by itself.

So if your hidden question is:

> “Which one lets me do something PINN-ish while still being course-aligned?”

then the best answers are **3**, then **1**.

---

## My honest recommendation

### If you want the most interesting PINN-adjacent project:
Choose **3**, but make it sharp:
- one PDE or ODE family,
- one conserved quantity / structure,
- compare soft penalties vs projection vs maybe architecture trick.

### If you want the strongest “Angelo + course” match:
Choose **5**.

### If you want the best compromise:
Do **1 + 5 together in a scoped way**:

> Sparse implicit differentiation for nonlinear projection layers in physics-informed ML.

That is actually a very strong project title.

It combines:
- the application story of **1**,
- the structural/numerical core of **5**,
- and it can still touch PINNs if you want.

---

## A concrete shortlist I’d propose

These are the three versions I think are best:

**A. Hard-constraint PINNs via differentiable projection layers**  
Most PINN-like.

**B. Sparse implicit differentiation for physics-constrained ML in Julia**  
Most aligned with your strengths.

**C. Structure-preserving neural ODE/PINN training**  
Most conceptually elegant.

My personal ranking for you:
**B**, then **C**, then **A**.

If you want, I can now turn **3** and **5** into two actual project pitches:
one in “professor-friendly proposal language,” and one in “what I would actually build week by week.”

---

# you asked

message time: 2026-03-19 10:31:01

Nice, in a paragraph... Could you explain what project 5 is about? Or give logistics?

---

# chatgpt response

Project 5 is about building a **Julia prototype for taking gradients through implicit scientific-computing procedures while automatically exploiting sparsity**. Concretely, many problems in scientific ML are defined not by an explicit formula `y = f(x)`, but by a condition like “`z` is the solution of a nonlinear system,” “`z` is the output of a constrained projection step,” or “`z` is the optimizer of some inner problem.” If you want to train a model around that, you need derivatives of `z` with respect to parameters, and the smart way to do that is often **implicit differentiation**: instead of backpropagating through every iteration of Newton/optimization, you differentiate the defining equations and solve a linear system involving a Jacobian or KKT matrix. The whole point of the project is that in SciML these matrices are often **sparse and structured**, so a naive dense autodiff pipeline wastes both time and memory; your project would ask how to automatically expose that sparsity and use sparse linear algebra to make gradients faster and more scalable.

Logistically, a very reasonable class-project version would be: first, pick **one family of implicit objects** such as nonlinear projection layers or steady-state solves; second, implement a small interface in Julia that defines the forward solve and then the implicit backward pass; third, compare three approaches on a few benchmark problems: unrolling all iterations, dense implicit differentiation, and sparse implicit differentiation. The deliverable would be a clean repo with a reusable differentiation wrapper, a few benchmark examples, timing/memory/accuracy plots, and a short report explaining when sparsity helps. So the project is not “build a PINN” directly, but rather “build the gradient engine that would make more advanced physics-constrained or implicit ML models practical.”

---

# you asked

message time: 2026-03-20 17:15:30

and which one out of all the projects we've brainstormed throughout this chat seems to fulfill this?

Propose a final project in 2-3 pages. Include references. You may work on
your own or in a team. The project can involve scientific machine learning, modeling,  
or HPC, but one way or another should involve Julia. It would be of interest 
to add AI or agents somehow. It is okay to begin a research project that you 
intend to continue after this class ends.

---

# chatgpt response

I’m mapping the class prompt against each idea by how easy it is to justify the keywords: scientific ML, modeling/HPC, Julia, and some AI angle.

The **cleanest fit** for that prompt is **Project 3: structure-preserving neural ODE/PINN training**.

It checks every box in the most obvious way: it is clearly **scientific machine learning**, it naturally involves **modeling** through ODEs/PDEs and physical structure, it is easy to implement in **Julia** using the SciML ecosystem, and the “AI” part is already built in because the model is neural. It is also very easy to justify in a 2–3 page proposal with references, because you can point to PINNs, neural ODEs, and physics-informed training as an existing literature, then say your contribution is to compare **soft penalties vs structure-preserving enforcement** such as conservation laws, positivity, or boundary-condition consistency. That makes it sound like a serious but still class-sized project, and it also leaves the door open to continue after the class.

The **second-best fit** is **Project 1: differentiable nonlinear projection layers for scientific ML**. This is a bit more original and closer to the PCFM spirit, and it still satisfies the prompt very well: scientific ML, modeling, Julia, and AI through its use inside physics-constrained learning. The only reason I put it slightly below Project 3 is that the “AI” story is a bit less immediate to a reader unless you explicitly say it will be used inside PINNs or other physics-informed neural models. So if you want the proposal that is **easiest to sell**, choose **3**; if you want the proposal that feels **slightly more distinctive and researchy**, choose **1**.

**Project 5** — sparse implicit differentiation — is also strong, but it fulfills the prompt a bit less directly. It definitely fits **Julia + scientific ML/HPC**, but the AI angle is more “infrastructure for advanced ML models” than “ML model itself,” so you would need more care in the writeup to explain why it belongs in this class. In other words: it is the most “Angelo-coded,” but **3** is the most obviously aligned with the assignment as written.

So my final answer is:

**Best fit for the class prompt:**  
**Structure-preserving neural ODE/PINN training**

**Best more-original alternative:**  
**Differentiable nonlinear projection layers for scientific ML**

A very strong compromise title would be:

**Hard-constraint PINNs via differentiable projection layers in Julia**

That one basically combines **3 + 1**, and I think it may actually be the strongest proposal of all.

---

# you asked

message time: 2026-03-20 17:18:18

But there is something i dont get from project 3... wouldnt all trainings be structure-preserving? isnt that the definition of PINN?? And how would it "standout" among the original (suggested by the professor / JuliaLab) projects???

---

# chatgpt response

I’m checking the uploaded proposals again so I can contrast your idea against the professor/JuliaLab-style projects concretely, not just abstractly.

No — that is exactly the subtle point: **a PINN is not automatically structure-preserving in the strong sense**.

A standard PINN usually means: you train a neural network by adding **physics-based loss terms** such as PDE residuals, boundary conditions, or initial conditions. That is **physics-informed**, but it does **not** usually guarantee exact preservation of deeper structure like conserved energy, mass, positivity, symplecticity, or divergence-free constraints. In practice, many PINNs only enforce these things **softly** through penalties, so the network may reduce the residual while still drifting in the quantity you actually care about. So “PINN” means “physics appears in the training objective,” whereas “structure-preserving” means “the learned dynamics or outputs satisfy some geometric/physical property by design or by a much stronger mechanism.” That is a real distinction, and it is exactly where Project 3 could live.

So for Project 3 to **stand out**, it should not be “I trained a PINN on a PDE.” That would be too generic. It should instead ask a sharper question like: **when do soft PINN penalties fail to preserve the right physics, and can hard or semi-hard enforcement help?** For example, you could compare three approaches on one benchmark PDE/ODE: ordinary PINN penalties, projection/correction onto the constraint set, and perhaps an architecture-level constraint. Then the contribution is not “I used a PINN,” but rather **a systematic study of hard vs soft structure enforcement in Julia SciML**, with quantitative comparisons on conservation error, training stability, and prediction quality. That would make it feel much more like a methods project.

Relative to the original suggested projects, this would stand out by being more **scientific-ML-facing** than the intrusive MOR project, which is mainly about symbolic automation and reduction passes in `ModelOrderReduction.jl`, and more **methodologically comparative** than the PCFM project, which is specifically about implementing nonlinear projection backends inside `PCFM.jl` for constrained generative inference. The MOR proposal is centered on symbolic transformations like Lift-and-Learn or PCE and integration into `ModelOrderReduction.jl`, with a clean implementation-and-PR workflow.   The PCFM proposal, by contrast, is already very focused on designing an `AbstractProjectionSolver` interface, exploiting block-diagonal sparsity, and benchmarking nonlinear projection backends with AD, GPU parallelism, Reactant, and MadNLP.  

Because of that, I think **plain Project 3 is not the one that stands out the most**. The version that really stands out is:

**“Hard-constraint PINNs via differentiable projection layers in Julia”**

That is basically **Project 3 + Project 1**, and maybe a touch of **5** if you want sparse implicit differentiation in the backward pass. Then your pitch becomes: standard PINNs enforce physics softly; I want to study whether adding differentiable projection/correction steps gives better physical fidelity and training behavior. That is distinctive, clearly Julia/SciML, clearly scientific ML, and it is different enough from both the MOR and PCFM suggestions while still being close to the course themes.

So my honest view is:

- **Project 3 alone:** valid, but risks sounding generic unless tightly framed.
- **Project 3 + 1:** much stronger and more distinctive.
- **Project 1 + 5:** strongest for your taste, but less obviously “PINN” on first read.

If your goal is both **fit the course** and **stand out**, I would not pitch Project 3 as “structure-preserving PINNs” by itself. I would pitch it as:

**“Beyond soft PINNs: differentiable hard-constraint enforcement for scientific machine learning in Julia.”**

That sounds much sharper.

---

# you asked

message time: 2026-03-20 17:22:55

I see, ok I like the project 3 + 1 vision/mindset... Could you now do a 2- or 3-page proposal detailing all as much as you can? Like obviously not the exact code we would write, but which things we would base on, what problem are trying to solve, previous work on it (either by Julia community packages or in general independent people), how would it contribute to the Julia ecosystem (if putting more time than just 1 semester), etc.

You could also use motivations of how to organize / convey ideas from the original proposals from the class. Could you confirm if you have access to them?

---

# chatgpt response

I do have access to the uploaded class proposals, and I’m using them as models for tone/scope while drafting yours. I’m pulling the relevant parts now so I can mirror the way they motivate the problem, methods, and ecosystem contribution.

I’ve got enough from the class proposals now. I’m adding a few primary-source references from the Julia/SciML ecosystem so the proposal can cite both the course context and the actual packages/literature it would build on.

Yes — I can access the uploaded class proposals, including the PCFM proposal, the intrusive MOR proposal, the TensorCore proposal, and the GeoOpt proposal, and I used them as models for tone and structure. In particular, I mirrored their pattern of: a clear motivation, a scoped methodology, an explicit Julia ecosystem story, concrete evaluation metrics, and a deliverable that could become a package contribution or PR beyond the class.    

Below is a proposal draft in the 2–3 page style.

---

# Beyond Soft PINNs: Differentiable Hard-Constraint Enforcement for Scientific Machine Learning in Julia

## Motivation and Problem Statement

Physics-informed neural networks (PINNs) and neural differential equation models have become a standard way to inject physical structure into machine learning by training neural networks against PDE or ODE residuals, boundary conditions, and initial conditions. In the Julia ecosystem, this paradigm is supported directly by NeuralPDE.jl for PINNs and by DiffEqFlux.jl for neural differential equations, while ModelingToolkit.jl provides a symbolic language for specifying the governing equations. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999118307125?utm_source=chatgpt.com))

However, a standard PINN is usually only **physics-informed in the loss function**, not necessarily **structure-preserving by construction**. In practice, conservation laws, positivity, normalization constraints, or nonlinear manifold constraints are often enforced only through soft penalties, and therefore can remain violated even when the training loss is low. This gap is one reason recent work has explored hard-constraint variants of PINNs and projection-based correction schemes. Lu et al. introduced hard-constrained PINNs for inverse design, while Physics-Constrained Flow Matching (PCFM) showed that nonlinear projection layers can enforce exact constraints during generative inference. ([SIAM Ebooks](https://epubs.siam.org/doi/10.1137/21M1397908?utm_source=chatgpt.com))

The goal of this project is to build a **Julia prototype for differentiable hard-constraint enforcement in scientific machine learning**, with a particular focus on PINNs and neural ODE/PDE models. The core question is:

**When does replacing soft constraint penalties by a differentiable projection/correction step improve physical fidelity, training stability, and downstream accuracy?**

This project is directly aligned with 18.337 because it combines scientific machine learning, modeling, automatic differentiation, optimization, and Julia-based implementation. It also has a clear AI component: the learned model is itself neural, and an optional extension is to add a lightweight agentic layer that proposes or schedules constraint-handling strategies during experiments. ([SCIML Docs](https://docs.sciml.ai/DiffEqDocs/latest/?utm_source=chatgpt.com))

## Proposed Idea

The proposed method is to augment a standard PINN or neural differential equation pipeline with a **ConstraintLayer** that maps a tentative model output `ẑ` to a corrected output `z*` satisfying one or more physical constraints. A generic form of this correction is:

minimize over z:  ||z - ẑ||²  
subject to:       c(z) = 0,  g(z) ≥ 0

where `c(z)` may encode mass conservation, energy conservation, boundary consistency, normalization, or other nonlinear physical constraints.

This layer can be placed in one of two settings:

1. **Output correction for PINNs**: after the network proposes a solution field over collocation points, project it onto a constrained set before evaluating the loss.
2. **Trajectory/state correction for neural ODEs**: after an integration step or rollout state is produced, project the state back to the physically admissible manifold.

The main novelty is not simply “using a PINN,” but building a reusable Julia interface for **hard or semi-hard constraint enforcement** and comparing it systematically against standard soft-penalty training. This takes inspiration from the PCFM project’s projection-backend abstraction, but moves the idea into the training-time regime for PINNs and neural differential equations rather than zero-shot inference in pretrained generative models.  ([arXiv](https://arxiv.org/abs/2506.04171?utm_source=chatgpt.com))

## Prior Work and Background

This project sits at the intersection of four existing lines of work.

First, classical PINNs, introduced by Raissi, Perdikaris, and Karniadakis, showed how PDE residuals and boundary conditions can be incorporated into neural network training. NeuralPDE.jl automated much of this workflow in Julia and demonstrated how symbolic PDE specifications can generate PINN training code, support multiple training strategies, and run on GPUs. ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999118307125?utm_source=chatgpt.com))

Second, neural ODEs and related continuous-time models use differential equation solvers as trainable layers, making differentiation through solves a first-class concern. DiffEqFlux.jl provides Julia support for these models and automatically selects appropriate differentiation strategies in many common cases. ([NeurIPS Papers](https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations?utm_source=chatgpt.com))

Third, optimization-as-a-layer work such as OptNet established that implicit constrained solves can be inserted into end-to-end differentiable models. That perspective is highly relevant here because a projection/correction step is itself an optimization layer. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/amos17a.html?utm_source=chatgpt.com))

Fourth, there is now clear motivation for **hard constraints** rather than only soft penalties. The hPINN work of Lu et al. demonstrated this in inverse design, while PCFM showed that nonlinear constraints can be imposed through iterative correction without sacrificing performance on learned physical distributions. These works support the view that exact or near-exact constraint enforcement is not redundant with standard physics-informed training. ([SIAM Ebooks](https://epubs.siam.org/doi/10.1137/21M1397908?utm_source=chatgpt.com))

## Methodology

The project will begin with a standard baseline in NeuralPDE.jl or DiffEqFlux.jl. ModelingToolkit.jl will be used to define the governing equations, domains, and symbolic structure of the problem. Optimization.jl will provide the solver interface for the projection step, allowing a clean initial implementation with Newton, quasi-Newton, or constrained optimizers. ([SCIML Docs](https://docs.sciml.ai/ModelingToolkit/?utm_source=chatgpt.com))

A scoped first version would support **one equality constraint family** and **one benchmark family**. For example:
- an ODE benchmark with energy preservation,
- or a PDE benchmark with mass conservation / boundary consistency.

The comparison will include at least three modes:
1. **Soft PINN baseline**: standard residual and boundary-condition penalties.
2. **Projected PINN / corrected neural DE**: apply the ConstraintLayer to enforce constraints.
3. **Hybrid method**: combine soft penalties with a lighter projection or augmented-Lagrangian-style correction.

For differentiation through the correction layer, the project can begin with unrolled differentiation through a fixed number of projection iterations. If time permits, a second version will use implicit differentiation through the optimality conditions, since the backward pass of a projection layer naturally reduces to solving Jacobian or KKT-type systems. That makes the project mathematically richer and closer to the 18.337 themes emphasized in the PCFM proposal.  ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/amos17a.html?utm_source=chatgpt.com))

## Julia Ecosystem Contribution

The most important software contribution would be a reusable module—tentatively `ConstraintLayers.jl` or `HardConstraintPINNs.jl`—that exposes a small abstraction for constrained correction inside scientific ML workflows. The design philosophy is similar to the class proposals: isolate one clean interface, connect it to the existing Julia stack, and validate it on representative examples. The proposed package would build on NeuralPDE.jl, DiffEqFlux.jl, ModelingToolkit.jl, and Optimization.jl rather than reimplementing an end-to-end solver from scratch.   ([SCIML Docs](https://docs.sciml.ai/DiffEqDocs/latest/?utm_source=chatgpt.com))

If this project were continued beyond one semester, there are at least three natural contributions to the Julia ecosystem:
- a generic constraint-layer API for PINN and neural ODE workflows,
- benchmark examples and documentation showing when hard constraints help,
- and possibly a PR or extension hook into existing SciML packages once the API stabilizes.

This is also where the project differs from the intrusive MOR proposal. That proposal contributes symbolic model-reduction passes to ModelOrderReduction.jl; by contrast, this project contributes a training-time constraint-enforcement mechanism for scientific ML models. The two projects share the same ecosystem mindset—symbolic specification, algorithmic implementation, and an eventual package-level contribution—but target different parts of the stack. 

## Evaluation Plan

The project will evaluate both **physical correctness** and **computational practicality**. The main metrics are:
- constraint violation after training or rollout,
- PDE/ODE residual error,
- solution error against a reference solver or known solution,
- training wall time,
- and training stability across random seeds or optimizer settings.

This metric design is inspired partly by the PCFM proposal, which explicitly evaluates constraint violation, wall time, and the effect of structure-aware correction on model quality. SciML Benchmarks also provide a natural reference point for the style of quantitative comparison expected in the Julia ecosystem.  ([SCIML Docs](https://docs.sciml.ai/SciMLBenchmarksOutput/?utm_source=chatgpt.com))

A successful semester-scale outcome would be:
1. one solid benchmark problem,
2. one reusable correction interface,
3. one clear empirical conclusion about when hard constraints outperform soft penalties.

A stretch goal would be a second benchmark or an implicit backward pass.

## Relation to the Course

This project fits the course prompt very naturally. It is a **scientific machine learning** project because the central model is a PINN or neural differential equation; it is also a **modeling** project because the constraints come from PDE/ODE structure; and it is a **Julia** project because the implementation would be built on the SciML stack. The AI component is already inherent in the neural model, and a small optional agentic extension could automatically propose optimizer schedules, constraint weights, or benchmark configurations, similar in spirit to the GeoOpt proposal’s use of an AI layer above a numerical simulator.  ([SCIML Docs](https://docs.sciml.ai/DiffEqDocs/latest/?utm_source=chatgpt.com))

Most importantly, it stands out from the suggested projects by occupying a distinct middle ground:
- unlike PCFM, it focuses on **training-time** constraint enforcement rather than inference-time correction for generative models,
- unlike intrusive MOR, it focuses on **learning with constraints** rather than symbolic reduction,
- and unlike a plain PINN project, it has a sharper methodological contribution than “apply NeuralPDE to a PDE.”  

## Expected Deliverables

By the end of the semester, I expect to produce:
- a Julia codebase implementing a first version of the ConstraintLayer,
- one benchmark study comparing soft vs hard constraint enforcement,
- a short report with methodology, results, and discussion,
- and a roadmap for how this could become a reusable package contribution after the course.

That scope is intentionally similar to the strongest class proposals: choose one focused track, implement it cleanly, and make the result extensible beyond the semester.  

## References

Raissi, Perdikaris, and Karniadakis, “Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations,” *Journal of Computational Physics* (2019). ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0021999118307125?utm_source=chatgpt.com))

Zubov et al., “NeuralPDE: Automating Physics-Informed Neural Networks (PINNs) with Error Approximations” (2021), and the NeuralPDE.jl documentation. ([arXiv](https://arxiv.org/abs/2107.09443?utm_source=chatgpt.com))

Chen, Rubanova, Bettencourt, and Duvenaud, “Neural Ordinary Differential Equations,” *NeurIPS* (2018), and the DiffEqFlux.jl neural ODE documentation. ([NeurIPS Papers](https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations?utm_source=chatgpt.com))

Amos and Kolter, “OptNet: Differentiable Optimization as a Layer in Neural Networks,” *ICML* (2017). ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v70/amos17a.html?utm_source=chatgpt.com))

Lu et al., “Physics-Informed Neural Networks with Hard Constraints for Inverse Design,” *SIAM Journal on Scientific Computing* (2021). ([SIAM Ebooks](https://epubs.siam.org/doi/10.1137/21M1397908?utm_source=chatgpt.com))

Utkarsh et al., “Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints” (2025), and the PCFM class proposal / PCFM.jl context. ([arXiv](https://arxiv.org/abs/2506.04171?utm_source=chatgpt.com)) 

ModelingToolkit.jl and Optimization.jl documentation. ([SCIML Docs](https://docs.sciml.ai/ModelingToolkit/?utm_source=chatgpt.com))

---

This is already in a submission-ready shape. The main thing I’d still decide is **which benchmark problem** you want to anchor it on, because that determines whether the proposal reads more “PINN-heavy” or more “neural ODE / constraint-layer-heavy.”