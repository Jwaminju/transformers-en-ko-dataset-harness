# Eval Alignment Review

- documents: 15
- chunks: 30

## accelerator_selection.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/accelerator_selection.md

### chunk 1
- source_chars: 644
- target_chars: 335
- length_ratio: 1.92

**Source**

During distributed training, you can specify the number and order of accelerators (CUDA, XPU, MPS, HPU, etc.) to use. This can be useful when you have accelerators with different computing power and you want to use the faster accelerator first. Or you could only use a subset of the available accelerators. The selection process works for both DistributedDataParallel and DataParallel. You don't need Accelerate or DeepSpeed integration. This guide will show you how to select the number of accelerators to use and the order to use them in. For example, if there are 4 accelerators and you only want to use the first 2, run the command below.

**Target**

분산 학습 과정에서 사용할 GPU의 개수와 순서를 정할 수 있습니다. 이 방법은 서로 다른 연산 성능을 가진 GPU가 있을 때 더 빠른 GPU를 우선적으로 사용하거나, 사용 가능한 GPU 중 일부만 선택하여 활용하고자 할 때 유용합니다. 이 선택 과정은 DistributedDataParallel과 DataParallel에서 모두 작동합니다. Accelerate나 DeepSpeed 통합은 필요하지 않습니다. 이 가이드는 사용할 GPU의 개수를 선택하는 방법과 사용 순서를 설정하는 방법을 설명합니다. 예를 들어, GPU가 4개 있고 그중 처음 2개만 사용하려는 경우, 아래 명령어를 실행하세요.

### chunk 2
- source_chars: 169
- target_chars: 131
- length_ratio: 1.29

**Source**

Use the --nprocpernode to select how many accelerators to use. Use --numprocesses to select how many accelerators to use. Use --numgpus to select how many GPUs to use.

**Target**

사용할 GPU 개수를 정하기 위해 --nprocpernode 옵션을 사용하세요. 사용할 GPU 개수를 정하기 위해 --numprocesses 옵션을 사용하세요. 사용할 GPU 개수를 정하기 위해 --numgpus 옵션을 사용하세요.

## add_new_model.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/add_new_model.md

### chunk 1
- source_chars: 392
- target_chars: 312
- length_ratio: 1.26

**Source**

> Try adding new models with a more modular approach first. This makes it significantly easier to contribute a model to Transformers! Many of the models in Transformers are contributed by developers and researchers. As an open-source first project, we're invested in empowering the community to actively and independently add more models. When you add a model to Transformers, you'll learn:

**Target**

Hugging Face Transformers 라이브러리는 커뮤니티 기여자들 덕분에 새로운 모델을 제공할 수 있는 경우가 많습니다. 하지만 이는 도전적인 프로젝트이며 Hugging Face Transformers 라이브러리와 구현할 모델에 대한 깊은 이해가 필요합니다. Hugging Face에서는 더 많은 커뮤니티 멤버가 모델을 적극적으로 추가할 수 있도록 지원하고자 하며, 이 가이드를 통해 PyTorch 모델을 추가하는 과정을 안내하고 있습니다 (PyTorch가 설치되어 있는지 확인해주세요). 이 과정을 진행하면 다음과 같은 내용을 이해하게 됩니다:

### chunk 2
- source_chars: 114
- target_chars: 114
- length_ratio: 1.0

**Source**

more about open-source best practices about Transformers' design principles how to efficiently test large models

**Target**

가장 인기 있는 딥러닝 라이브러리의 설계 원칙을 이해합니다. black, ruff, make fix-repo와 같은 Python 유틸리티를 통합하여 깔끔하고 가독성 있는 코드를 작성하는 방법을 배웁니다.

## add_new_pipeline.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/add_new_pipeline.md

### chunk 1
- source_chars: 299
- target_chars: 175
- length_ratio: 1.71

**Source**

Make [Pipeline] your own by subclassing it and implementing a few methods. Share the code with the community on the Hub and register the pipeline with Transformers so that everyone can quickly and easily use it. This guide will walk you through the process of adding a new pipeline to Transformers.

**Target**

이 가이드에서는 사용자 정의 파이프라인을 어떻게 생성하고 허브에 공유하거나 🤗 Transformers 라이브러리에 추가하는 방법을 살펴보겠습니다. 먼저 파이프라인이 수용할 수 있는 원시 입력을 결정해야 합니다. 문자열, 원시 바이트, 딕셔너리 또는 가장 원하는 입력일 가능성이 높은 것이면 무엇이든 가능합니다.

### chunk 2
- source_chars: 153
- target_chars: 144
- length_ratio: 1.06

**Source**

At a minimum, you only need to provide [Pipeline] with an appropriate input for a task. This is also where you should begin when designing your pipeline.

**Target**

이 입력을 가능한 한 순수한 Python 형식으로 유지해야 (JSON을 통해 다른 언어와도) 호환성이 좋아집니다. 이것이 전처리(preprocess) 파이프라인의 입력(inputs)이 될 것입니다. inputs와 같은 정책을 따르고, 간단할수록 좋습니다.

## cache_explanation.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/cache_explanation.md

### chunk 1
- source_chars: 594
- target_chars: 282
- length_ratio: 2.11

**Source**

Imagine you're having a conversation with someone, and instead of remembering what they previously said, they have to start from scratch every time you respond. This would be slow and inefficient, right? You can extend this analogy to transformer models. Autoregressive model generation can be slow because it makes a prediction one token at a time. Each new prediction is dependent on all the previous context. To predict the 1000th token, the model requires information from the previous 999 tokens. The information is represented as matrix multiplications across the token representations.

**Target**

누군가와 대화를 나누고 있는데, 상대방이 이전에 했던 말을 기억하지 못하고 당신이 대답할 때마다 처음부터 다시 시작해야 한다고 상상해 보세요. 이는 느리고 비효율적이겠죠? 이 비유를 트랜스포머 모델에도 적용할 수 있습니다. 자기회귀 모델의 생성은 한 번에 하나의 토큰씩 예측하기 때문에 느릴 수 있습니다. 각각의 새로운 예측은 이전의 모든 문맥에 의존합니다. 1000번째 토큰을 예측하려면, 모델은 이전 999개 토큰의 정보가 필요합니다. 이 정보는 각 토큰 표현들 사이의 행렬 곱을 통해 표현됩니다.

### chunk 2
- source_chars: 598
- target_chars: 290
- length_ratio: 2.06

**Source**

To predict the 1001th token, you need the same information from the previous 999 tokens in addition to any information from the 1000th token. This is a lot of matrix multiplications a model has to compute over and over for each token! A key-value (KV) cache eliminates this inefficiency by storing kv pairs derived from the attention layers of previously processed tokens. The stored kv pairs are retrieved from the cache and reused for subsequent tokens, avoiding the need to recompute. > Caching should only be used for inference. It may cause unexpected errors if it's enabled during training.

**Target**

1001번째 토큰을 예측하려면, 이전 999개 토큰의 동일한 정보에 더하여 1000번째 토큰의 정보도 필요합니다. 이렇게 되면 토큰마다 모델은 반복적으로 많은 행렬 연산을 수행해야 합니다! 이러한 비효율성을 제거하기 위해 KV 캐시(Key-Value Cache)를 사용합니다. 어텐션 레이어에서 이전에 처리한 토큰으로부터 얻은 키와 값 쌍을 저장해두고, 이후 토큰 예측 시 이를 재사용하여 연산을 줄이는 방식입니다. > 캐싱은 추론에만 사용해야 합니다. 학습 중에 활성화되면 예상치 못한 오류가 발생할 수 있습니다.

## chat_extras.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/chat_extras.md

### chunk 1
- source_chars: 582
- target_chars: 322
- length_ratio: 1.81

**Source**

Chat models are commonly trained with support for "function-calling" or "tool-use". Tools are functions supplied by the user, which the model can choose to call as part of its response. For example, models could have access to a calculator tool to perform arithmetic without having to perform the computation internally. This guide will demonstrate how to define tools, how to pass them to a chat model, and how to handle the model's output when it calls a tool. When a model supports tool-use, pass functions to the tools argument of [~PreTrainedTokenizerBase.applychattemplate].

**Target**

[~PreTrainedTokenizerBase.applychattemplate] 메소드는 채팅 메시지 외에도 문자열, 리스트, 딕셔너리 등 거의 모든 종류의 추가 인수 타입을 지원합니다. 이를 통해 다양한 사용 상황에서 채팅 템플릿을 활용할 수 있습니다. 이 가이드에서는 도구 및 검색 증강 생성(RAG)과 함께 채팅 템플릿을 사용하는 방법을 보여드립니다. 도구는 대규모 언어 모델(LLM)이 특정 작업을 수행하기 위해 호출할 수 있는 함수입니다. 이는 실시간 정보, 계산 도구 또는 대규모 데이터베이스 접근 등을 통해 대화형 에이전트의 기능을 확장하는 강력한 방법입니다.

### chunk 2
- source_chars: 308
- target_chars: 179
- length_ratio: 1.72

**Source**

The tools are passed as either a JSON schema or Python functions. If you pass Python functions, the arguments, argument types, and function docstring are parsed in order to generate the JSON schema automatically. Although passing Python functions is very convenient, the parser can only handle Google-style

**Target**

함수의 인수는 함수 헤더에 타입 힌트를 포함해야 합니다(Args 블록에는 포함하지 마세요). 함수에는 Google 스타일 의 독스트링(docstring)이 포함되어야 합니다. 함수에 반환 타입과 Returns 블록을 포함할 수 있지만, 도구를 활용하는 대부분의 모델에서 이를 사용하지 않기 때문에 무시할 수 있습니다.

## community.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/community.md

### chunk 1
- source_chars: 115
- target_chars: 88
- length_ratio: 1.31

**Source**

This page regroups resources around 🤗 Transformers developed by the community. | Resource | Description | Author |

**Target**

이 페이지는 커뮤니티에서 개발한 🤗 Transformers 리소스를 재구성한 페이지입니다. |:----------|:-------------|------:|

### chunk 2
- source_chars: 434
- target_chars: 377
- length_ratio: 1.15

**Source**

|:----------|:-------------|------:| | Hugging Face Transformers Glossary Flashcards | A set of flashcards based on the Transformers Docs Glossary that has been put into a form which can be easily learned/revised using Anki an open source, cross platform app specifically designed for long term knowledge retention. See this Introductory video on how to use the flashcards. | Darigov Research | | Notebook | Description | Author | |

**Target**

| Hugging Face Transformers 용어집 플래시카드 | Transformers 문서 용어집을 기반으로 한 플래시카드 세트로, 지식을 장기적으로 유지하기 위해 특별히 설계된 오픈소스 크로스 플랫폼 앱인 Anki를 사용하여 쉽게 학습/수정할 수 있는 형태로 제작되었습니다. 플래시카드 사용법에 대한 소개 동영상을 참조하세요. | Darigov 리서치 | |:----------|:-------------|:-------------|------:| | 가사를 생성하기 위해 사전훈련된 트랜스포머를 미세 조정하기 | GPT-2 모델을 미세 조정하여 좋아하는 아티스트의 스타일로 가사를 생성하는 방법 | Aleksey Korshuk | Open In Colab |

## custom_models.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/custom_models.md

### chunk 1
- source_chars: 251
- target_chars: 173
- length_ratio: 1.45

**Source**

Transformers models are designed to be customizable. A models code is fully contained in the model subfolder of the Transformers repository. Each folder contains a modeling.py and a configuration.py file. Copy these files to start customizing a model.

**Target**

🤗 Transformers 라이브러리는 쉽게 확장할 수 있도록 설계되었습니다. 모든 모델은 추상화 없이 저장소의 지정된 하위 폴더에 완전히 코딩되어 있으므로, 손쉽게 모델링 파일을 복사하고 필요에 따라 조정할 수 있습니다. 완전히 새로운 모델을 만드는 경우에는 처음부터 시작하는 것이 더 쉬울 수 있습니다.

### chunk 2
- source_chars: 229
- target_chars: 216
- length_ratio: 1.06

**Source**

> It may be easier to start from scratch if you're creating an entirely new model. But for models that are very similar to an existing one in Transformers, it is faster to reuse or subclass the same configuration and model class.

**Target**

이 튜토리얼에서는 Transformers 내에서 사용할 수 있도록 사용자 정의 모델과 구성을 작성하는 방법과 🤗 Transformers 라이브러리에 없는 경우에도 누구나 사용할 수 있도록 (의존성과 함께) 커뮤니티에 공유하는 방법을 배울 수 있습니다. timm 라이브러리의 ResNet 클래스를 [PreTrainedModel]로 래핑한 ResNet 모델을 예로 모든 것을 설명합니다.

## deepspeed.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/deepspeed.md

### chunk 1
- source_chars: 398
- target_chars: 399
- length_ratio: 1.0

**Source**

DeepSpeed is designed to optimize distributed training for large models with data, model, pipeline, and even a combination of all three parallelism strategies to provide better memory efficiency and faster training speeds. This is achieved with the Zero Redundancy Optimizer (ZeRO) which consists of three stages. | 1 | partition optimizer states | | 2 | partition optimizer and gradient states |

**Target**

DeepSpeed는 분산 학습 메모리를 효율적이고 빠르게 만드는 PyTorch 최적화 라이브러리입니다. 그 핵심은 대규모 모델을 규모에 맞게 훈련할 수 있는 Zero Redundancy Optimizer(ZeRO)입니다. ZeRO는 여러 단계로 작동합니다: GPU가 제한된 환경에서 ZeRO는 최적화 메모리와 계산을 GPU에서 CPU로 오프로드하여 단일 GPU에 대규모 모델을 장착하고 훈련할 수 있습니다. DeepSpeed는 모든 ZeRO 단계 및 오프로딩을 위해 Transformers [Trainer] 클래스와 통합되어 있습니다. 구성 파일을 제공하거나 제공된 템플릿을 사용하기만 하면 됩니다. 추론의 경우, Transformers는 대용량 모델을 가져올 수 있으므로 ZeRO-3 및 오프로딩을 지원합니다.

### chunk 2
- source_chars: 507
- target_chars: 415
- length_ratio: 1.22

**Source**

| 3 | partition optimizer, gradient, and parameters | Each stage progressively saves more memory, allowing really large models to fit and train on a single GPU. All ZeRO stages, offloading optimizer memory and computations from the GPU to the CPU are integrated with [Trainer]. Provide a config file or one of the example templates to [Trainer] to enable DeepSpeed features. This guide walks you through setting up a DeepSpeed config file, how to enable its features in [Trainer], and deploy for training.

**Target**

이 가이드에서는 DeepSpeed 트레이닝을 배포하는 방법, 활성화할 수 있는 기능, 다양한 ZeRO 단계에 대한 구성 파일 설정 방법, 오프로딩, 추론 및 [Trainer] 없이 DeepSpeed를 사용하는 방법을 안내해 드립니다. DeepSpeed는 PyPI 또는 Transformers에서 설치할 수 있습니다(자세한 설치 옵션은 DeepSpeed 설치 상세사항 또는 GitHub README를 참조하세요). DeepSpeed를 설치하는 데 문제가 있는 경우 DeepSpeed CUDA 설치 가이드를 확인하세요. DeepSpeed에는 pip 설치 가능한 PyPI 패키지로 설치할 수 있지만, 하드웨어에 가장 잘 맞고 PyPI 배포판에서는 제공되지 않는 1비트 Adam과 같은 특정 기능을 지원하려면 소스에서 설치하기를 적극 권장합니다.

## generation_strategies.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/generation_strategies.md

### chunk 1
- source_chars: 454
- target_chars: 242
- length_ratio: 1.88

**Source**

A decoding strategy informs how a model should select the next generated token. There are many types of decoding strategies, and choosing the appropriate one has a significant impact on the quality of the generated text. This guide will help you understand the different decoding strategies available in Transformers and how and when to use them. These are well established decoding methods, and should be your starting point for text generation tasks.

**Target**

텍스트 생성은 개방형 텍스트 작성, 요약, 번역 등 다양한 자연어 처리(NLP) 작업에 필수적입니다. 이는 또한 음성-텍스트 변환, 시각-텍스트 변환과 같이 텍스트를 출력으로 하는 여러 혼합 모달리티 응용 프로그램에서도 중요한 역할을 합니다. 텍스트 생성을 가능하게 하는 몇몇 모델로는 GPT2, XLNet, OpenAI GPT, CTRL, TransformerXL, XLM, Bart, T5, GIT, Whisper 등이 있습니다.

### chunk 2
- source_chars: 750
- target_chars: 316
- length_ratio: 2.37

**Source**

Greedy search is the default decoding strategy. It selects the next most likely token at each step. Unless specified in [GenerationConfig], this strategy generates a maximum of 20 new tokens. Greedy search works well for tasks with relatively short outputs where creativity is not a priority. However, it breaks down when generating longer sequences because it begins to repeat itself. Sampling, or multinomial sampling, randomly selects a token based on the probability distribution over the entire model's vocabulary (as opposed to the most likely token, as in greedy search). This means every token with a non-zero probability has a chance to be selected. Sampling strategies reduce repetition and can generate more creative and diverse outputs.

**Target**

[~generation.GenerationMixin.generate] 메서드를 활용하여 다음과 같은 다양한 작업들에 대해 텍스트 결과물을 생성하는 몇 가지 예시를 살펴보세요: generate 메소드에 입력되는 값들은 모델의 데이터 형태에 따라 달라집니다. 이 값들은 AutoTokenizer나 AutoProcessor와 같은 모델의 전처리 클래스에 의해 반환됩니다. 모델의 전처리 장치가 하나 이상의 입력 유형을 생성하는 경우, 모든 입력을 generate()에 전달해야 합니다. 각 모델의 전처리 장치에 대해서는 해당 모델의 문서에서 자세히 알아볼 수 있습니다.

## glossary.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/glossary.md

### chunk 1
- source_chars: 284
- target_chars: 197
- length_ratio: 1.44

**Source**

This glossary defines general machine learning and 🤗 Transformers terms to help you better understand the The attention mask is an optional argument used when batching sequences together. This argument indicates to the model which tokens should be attended to, and which should not.

**Target**

이 용어집은 전반적인 머신러닝 및 🤗 Transformers 관련 용어를 정의하여 문서를 더 잘 이해하는 데 도움을 줍니다. 어텐션 마스크(attention mask)는 여러 시퀀스를 배치(batch)로 처리할 때 사용되는 선택적 인자입니다. 이 인자는 모델에게 어떤 토큰에 주의를 기울여야 하는지, 그리고 어떤 토큰은 무시해야 하는지를 알려줍니다.

### chunk 2
- source_chars: 208
- target_chars: 219
- length_ratio: 0.95

**Source**

For example, consider these two sequences: The encoded versions have different lengths: Therefore, we can't put them together in the same tensor as-is. The first sequence needs to be padded up to the length

**Target**

예를 들어, 다음 두 개의 시퀀스가 있다고 가정해 봅시다: 따라서 이 두 시퀀스를 그대로 하나의 텐서에 넣을 수는 없습니다. 첫 번째 시퀀스를 두 번째 길이에 맞춰 패딩 하거나, 반대로 두 번째 시퀀스를 첫 번째 길이에 맞춰 잘라내야 합니다. 첫 번째 경우에는 ID 목록이 패딩 인덱스로 확장됩니다. 이렇게 패딩을 적용하려면 토크나이저에 리스트를 전달하고 다음과 같이 요청할 수 있습니다:

## how_to_hack_models.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/how_to_hack_models.md

### chunk 1
- source_chars: 785
- target_chars: 451
- length_ratio: 1.74

**Source**

Another way to customize a model is to modify their components, rather than writing a new model entirely, allowing you to tailor a model to your specific use case. For example, you can add new layers or optimize the attention mechanism of an architecture. Customizations are applied directly to a Transformers model so that you can continue to use features such as [Trainer], [PreTrainedModel], and the PEFT library. This guide will show you how to customize a models attention mechanism in order to apply Low-Rank Adaptation (LoRA) to it. > The clearimportcache utility is very useful when you're iteratively modifying and developing model code. It removes all cached Transformers modules and allows Python to reload the modified code without constantly restarting your environment.

**Target**

모델을 완전히 새로 작성하는 대신 구성 요소를 수정하여 모델을 맞춤 설정하는 방법이 있습니다. 이 방법으로 모델을 특정 사용 사례에 맞게 모델을 조정할 수 있습니다. 예를 들어, 새로운 레이어를 추가하거나 아키텍처의 어텐션 메커니즘을 최적화할 수 있습니다. 이러한 맞춤 설정은 트랜스포머 모델에 직접 적용되므로, [Trainer], [PreTrainedModel] 및 PEFT 라이브러리와 같은 기능을 계속 사용할 수 있습니다. 이 가이드에서는 모델의 어텐션 메커니즘을 맞춤 설정하여 Low-Rank Adaptation (LoRA)를 적용하는 방법을 설명합니다. > 모델 코드를 반복적으로 수정하고 개발할 때 clearimportcache 유틸리티가 매우 유용합니다. 이 기능은 캐시된 모든 트랜스포머 모듈을 제거하여 Python이 환경을 재시작하지 않고도 수정된 코드를 다시 가져올 수 있도록 합니다.

### chunk 4
- source_chars: 117
- target_chars: 91
- length_ratio: 1.29

**Source**

Load the model with [~PreTrainedModel.frompretrained]. With separate q, k, and v projections, apply LoRA to q and v.

**Target**

[~PreTrainedModel.frompretrained]로 모델을 가져오세요. 분리된 q, k, v 프로젝션을 사용할 때 , q와 v에 LoRA를 적용합니다.

## hpo_train.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/hpo_train.md

### chunk 1
- source_chars: 617
- target_chars: 302
- length_ratio: 2.04

**Source**

Hyperparameter search discovers an optimal set of hyperparameters that produces the best model performance. [Trainer] supports several hyperparameter search backends - Optuna, Weights & Biases, Ray Tune - through [~Trainer.hyperparametersearch] to optimize an objective or even multiple objectives. This guide will go over how to set up a hyperparameter search for each of the backends. To use [~Trainer.hyperparametersearch], you need to create a modelinit function. This function includes basic model information (arguments and configuration) because it needs to be reinitialized for each search trial in the run.

**Target**

🤗 Transformers에서는 🤗 Transformers 모델을 학습시키는데 최적화된 [Trainer] 클래스를 제공하기 때문에, 사용자는 직접 훈련 루프를 작성할 필요 없이 더욱 간편하게 학습을 시킬 수 있습니다. 또한, [Trainer]는 하이퍼파라미터 탐색을 위한 API를 제공합니다. 이 문서에서 이 API를 활용하는 방법을 예시와 함께 보여드리겠습니다. [Trainer]는 현재 아래 4가지 하이퍼파라미터 탐색 백엔드를 지원합니다: 하이퍼파라미터 탐색 백엔드로 사용하기 전에 아래의 명령어를 사용하여 라이브러리들을 설치하세요.

### chunk 2
- source_chars: 394
- target_chars: 166
- length_ratio: 2.37

**Source**

> The modelinit function is incompatible with the optimizers parameter. Subclass [Trainer] and override the [~Trainer.createoptimizerandscheduler] method to create a custom optimizer and scheduler. An example modelinit function is shown below. Pass modelinit to [Trainer] along with everything else you need for training. Then you can call [~Trainer.hyperparametersearch] to start the search.

**Target**

하이퍼파라미터 탐색 공간을 정의하세요. 하이퍼파라미터 탐색 백엔드마다 서로 다른 형식이 필요합니다. optuna의 경우, 해당 objectparameter 문서를 참조하여 아래와 같이 작성하세요: raytune의 경우, 해당 objectparameter 문서를 참조하여 아래와 같이 작성하세요:

## image_processors.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/image_processors.md

### chunk 1
- source_chars: 399
- target_chars: 335
- length_ratio: 1.19

**Source**

Image processors convert images into pixel values, tensors that represent image colors and size. The pixel values are inputs to a vision model. To ensure a pretrained model receives the correct input, an image processor can perform the following operations to make sure an image is exactly like the images a model was pretrained on. center-crop or resize an image normalize or rescale pixel values

**Target**

이미지 프로세서는 이미지를 픽셀 값, 즉 이미지의 색상과 크기를 나타내는 텐서로 변환합니다. 이 픽셀 값은 비전 모델의 입력으로 사용됩니다. 이때 사전 학습된 모델이 새로운 이미지를 올바르게 인식하려면 입력되는 이미지의 형식이 학습 당시 사용했던 데이터와 똑같아야 합니다. 이미지 프로세서는 다음과 같은 작업을 통해 이미지 형식을 통일시켜주는 역할을 합니다. 이미지 크기를 조절하는 [~BaseImageProcessor.centercrop] 픽셀 값을 정규화하는 [~BaseImageProcessor.normalize] 또는 크기를 재조정하는 [~BaseImageProcessor.rescale]

### chunk 2
- source_chars: 589
- target_chars: 371
- length_ratio: 1.59

**Source**

Use [~ImageProcessingMixin.frompretrained] to load an image processors configuration (image size, whether to normalize and rescale, etc.) from a vision model on the Hugging Face Hub or local directory. The configuration for each pretrained model is saved in a preprocessorconfig.json file. Pass an image to the image processor to transform it into pixel values, and set returntensors="pt" to return PyTorch tensors. Feel free to print out the inputs to see what the image looks like as a tensor. This guide covers the image processor class and how to preprocess images for vision models.

**Target**

Hugging Face Hub나 로컬 디렉토리에 있는 비전 모델에서 이미지 프로세서의 설정(이미지 크기, 정규화 및 리사이즈 여부 등)을 불러오려면 [~ImageProcessingMixin.frompretrained]를 사용하세요. 각 사전 학습된 모델의 설정은 preprocessorconfig.json 파일에 저장되어 있습니다. 이미지를 이미지 프로세서에 전달하여 픽셀 값으로 변환하고, returntensors="pt" 를 설정하여 PyTorch 텐서를 반환받으세요. 이미지가 텐서로 어떻게 보이는지 궁금하다면 입력값을 한번 출력해보시는걸 추천합니다! 이 가이드에서는 이미지 프로세서 클래스와 비전 모델을 위한 이미지 전처리 방법에 대해 다룰 예정입니다.

## installation.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/installation.md

### chunk 1
- source_chars: 285
- target_chars: 205
- length_ratio: 1.39

**Source**

Transformers works with PyTorch. It has been tested on Python 3.10+ and PyTorch 2.4+. uv is an extremely fast Rust-based Python package and project manager and requires a virtual environment by default to manage different projects and avoids compatibility issues between dependencies.

**Target**

🤗 Transformers를 사용 중인 딥러닝 라이브러리에 맞춰 설치하고, 캐시를 구성하거나 선택적으로 오프라인에서도 실행할 수 있도록 🤗 Transformers를 설정하는 방법을 배우겠습니다. 🤗 Transformers는 Python 3.10+ 및 PyTorch 2.4+에서 테스트되었습니다. 딥러닝 라이브러리를 설치하려면 아래 링크된 공식 사이트를 참고해주세요.

### chunk 2
- source_chars: 223
- target_chars: 211
- length_ratio: 1.06

**Source**

It can be used as a drop-in replacement for pip, but if you prefer to use pip, remove uv from the commands below. > Refer to the uv installation docs to install uv. Create a virtual environment to install Transformers in.

**Target**

🤗 Transformers를 가상 환경에 설치하는 것을 추천드립니다. Python 가상 환경에 익숙하지 않다면, 이 가이드를 참고하세요. 가상 환경을 사용하면 서로 다른 프로젝트들을 보다 쉽게 관리할 수 있고, 의존성 간의 호환성 문제를 방지할 수 있습니다. 먼저 프로젝트 디렉토리에서 가상 환경을 만들어 줍니다. 가상 환경을 활성화해주세요. Linux나 MacOS의 경우:

## internal/generation_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/generation_utils.md

### chunk 2
- source_chars: 273
- target_chars: 182
- length_ratio: 1.5

**Source**

The output of [~generation.GenerationMixin.generate] is an instance of a subclass of [~utils.ModelOutput]. This output is a data structure containing all the information returned by [~generation.GenerationMixin.generate], but that can also be used as tuple or dictionary.

**Target**

[~generation.GenerationMixin.generate]의 출력은 [~utils.ModelOutput]의 하위 클래스의 인스턴스입니다. 이 출력은 [~generation.GenerationMixin.generate]에서 반환되는 모든 정보를 포함하는 데이터 구조체이며, 튜플 또는 딕셔너리로도 사용할 수 있습니다.

### chunk 3
- source_chars: 218
- target_chars: 111
- length_ratio: 1.96

**Source**

The generationoutput object is a [~generation.GenerateDecoderOnlyOutput], as we can see in the documentation of that class below, it means it has the following attributes: sequences: the generated sequences of tokens

**Target**

generationoutput 객체는 [~generation.GenerateDecoderOnlyOutput]입니다. 아래 문서에서 확인할 수 있듯이, 이 클래스는 다음과 같은 속성을 가지고 있습니다:
