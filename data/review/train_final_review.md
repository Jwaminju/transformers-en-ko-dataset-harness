# Train Final Review

- documents: 24
- chunks: 40

## internal/audio_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/audio_utils.md

### chunk 1
- source_chars: 329
- target_chars: 223
- length_ratio: 1.48

**Source**

This page lists all the utility functions that can be used by the audio [FeatureExtractor] in order to compute special features from a raw audio using common algorithms such as Short Time Fourier Transform or log mel spectrogram. Most of those are only useful if you are studying the code of the audio processors in the library.

**Target**

이 페이지는 오디오 [FeatureExtractor]가 단시간 푸리에 변환(Short Time Fourier Transform) 또는 로그 멜 스펙트로그램(log mel spectrogram)과 같은 일반적인 알고리즘을 사용하여 원시 오디오에서 특수한 특성을 계산하는 데 사용할 수 있는 유틸리티 함수들을 나열합니다. 이 함수들 대부분은 라이브러리 내 오디오 처리 코드를 연구할 때에만 유용합니다.

## internal/file_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/file_utils.md

### chunk 1
- source_chars: 222
- target_chars: 138
- length_ratio: 1.61

**Source**

This page lists all of Transformers general utility functions that are found in the file utils.py. Most of those are only useful if you are studying the general code in the library. utils.addstartdocstringstomodelforward

**Target**

이 페이지는 utils.py 파일에 있는 Transformers의 일반 유틸리티 함수들을 나열합니다. 이 함수들 대부분은 라이브러리의 일반적인 코드를 연구할 때만 유용합니다. utils.addstartdocstringstomodelforward

## internal/modeling_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/modeling_utils.md

### chunk 1
- source_chars: 256
- target_chars: 135
- length_ratio: 1.9

**Source**

This page lists all the custom layers used by the library, as well as the utility functions and classes it provides for modeling. Most of those are only useful if you are studying the code of the models in the library. pytorchutils.applychunkingtoforward

**Target**

이 페이지는 라이브러리에서 사용되는 사용자 정의 레이어와 모델링을 위한 유틸리티 함수들을 나열합니다. 이 함수들 대부분은 라이브러리 내의 모델 코드를 연구할 때만 유용합니다. pytorchutils.applychunkingtoforward

## internal/pipelines_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/pipelines_utils.md

### chunk 1
- source_chars: 216
- target_chars: 140
- length_ratio: 1.54

**Source**

This page lists all the utility functions the library provides for pipelines. Most of those are only useful if you are studying the code of the models in the library. pipelines.ZeroShotClassificationArgumentHandler

**Target**

이 페이지는 라이브러리에서 파이프라인을 위해 제공하는 모든 유틸리티 함수들을 나열합니다. 이 함수들 대부분은 라이브러리 내 모델의 코드를 연구할 때만 유용합니다. pipelines.ZeroShotClassificationArgumentHandler

### chunk 2
- source_chars: 100
- target_chars: 100
- length_ratio: 1.0

**Source**

pipelines.CsvPipelineDataFormat pipelines.JsonPipelineDataFormat pipelines.PipedPipelineDataFormat

**Target**

pipelines.CsvPipelineDataFormat pipelines.JsonPipelineDataFormat pipelines.PipedPipelineDataFormat

## internal/time_series_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/time_series_utils.md

### chunk 1
- source_chars: 289
- target_chars: 145
- length_ratio: 1.99

**Source**

This page lists all the utility functions and classes that can be used for Time Series based models. Most of those are only useful if you are studying the code of the time series models or you wish to add to the collection of distributional output classes. timeseriesutils.StudentTOutput

**Target**

이 페이지는 시계열 기반 모델에서 사용할 수 있는 유틸리티 함수와 클래스들을 나열합니다. 이 함수들 대부분은 시계열 모델의 코드를 연구하거나 분포 출력 클래스의 컬렉션에 추가하려는 경우에만 유용합니다. timeseriesutils.StudentTOutput

## internal/trainer_utils.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/internal/trainer_utils.md

### chunk 1
- source_chars: 184
- target_chars: 132
- length_ratio: 1.39

**Source**

This page lists all the utility functions used by [Trainer]. Most of those are only useful if you are studying the code of the Trainer in the library. trainercallback.CallbackHandler

**Target**

이 페이지는 [Trainer]에서 사용되는 모든 유틸리티 함수들을 나열합니다. 이 함수들 대부분은 라이브러리에 있는 Trainer 코드를 자세히 알아보고 싶을 때만 유용합니다. trainercallback.CallbackHandler

## main_classes/configuration.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/configuration.md

### chunk 1
- source_chars: 255
- target_chars: 164
- length_ratio: 1.55

**Source**

The base class [PreTrainedConfig] implements the common methods for loading/saving a configuration either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's AWS S3 repository).

**Target**

기본 클래스 [PreTrainedConfig]는 로컬 파일이나 디렉토리, 또는 라이브러리에서 제공하는 사전 학습된 모델 구성(HuggingFace의 AWS S3 저장소에서 다운로드됨)으로부터 구성을 불러오거나 저장하는 공통 메서드를 구현합니다. 각 파생 구성 클래스는 모델별 특성을 구현합니다.

### chunk 2
- source_chars: 200
- target_chars: 110
- length_ratio: 1.82

**Source**

Each derived config class implements model specific attributes. Common attributes present in all config classes are: hiddensize, numattentionheads, and numhiddenlayers. Text models further implement:

**Target**

모든 구성 클래스에 존재하는 공통 속성은 다음과 같습니다: hiddensize, numattentionheads, numhiddenlayers. 텍스트 모델은 추가로 vocabsize를 구현합니다.

## main_classes/data_collator.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/data_collator.md

### chunk 1
- source_chars: 286
- target_chars: 283
- length_ratio: 1.01

**Source**

Data collators are objects that will form a batch by using a list of dataset elements as input. These elements are of the same type as the elements of traindataset or evaldataset. To be able to build batches, data collators may apply some processing (like padding). Some of them (like

**Target**

데이터 콜레이터는 데이터셋 요소들의 리스트를 입력으로 사용하여 배치를 형성하는 객체입니다. 이러한 요소들은 traindataset 또는 evaldataset의 요소들과 동일한 타입 입니다. 배치를 구성하기 위해, 데이터 콜레이터는 (패딩과 같은) 일부 처리를 적용할 수 있습니다. [DataCollatorForLanguageModeling]과 같은 일부 콜레이터는 형성된 배치에 (무작위 마스킹과 같은) 일부 무작위 데이터 증강도 적용합니다. 사용 예시는 예제 스크립트나 예제 노트북에서 찾을 수 있습니다.

### chunk 2
- source_chars: 211
- target_chars: 119
- length_ratio: 1.77

**Source**

[DataCollatorForLanguageModeling]) also apply some random data augmentation (like random masking) Examples of use can be found in the example scripts or example notebooks. data.datacollator.defaultdatacollator

**Target**

data.datacollator.defaultdatacollator data.datacollator.DefaultDataCollator data.datacollator.DataCollatorWithPadding

## main_classes/feature_extractor.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/feature_extractor.md

### chunk 1
- source_chars: 334
- target_chars: 312
- length_ratio: 1.07

**Source**

A feature extractor is in charge of preparing input features for audio models. This includes feature extraction from sequences, e.g., pre-processing audio files to generate Log-Mel Spectrogram features, and conversion to NumPy and PyTorch tensors. featureextractionutils.FeatureExtractionMixin imageutils.ImageFeatureExtractionMixin

**Target**

특성 추출기는 오디오 또는 비전 모델을 위한 입력 특성을 준비하는 역할을 합니다. 여기에는 시퀀스에서 특성을 추출하는 작업(예를 들어, 오디오 파일을 전처리하여 Log-Mel 스펙트로그램 특성을 생성하는 것), 이미지에서 특성을 추출하는 작업(예를 들어, 이미지 파일을 자르는 것)이 포함됩니다. 뿐만 아니라 패딩, 정규화 및 NumPy, PyTorch, TensorFlow 텐서로의 변환도 포함됩니다. featureextractionutils.FeatureExtractionMixin imageutils.ImageFeatureExtractionMixin

## main_classes/model.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/model.md

### chunk 1
- source_chars: 322
- target_chars: 316
- length_ratio: 1.02

**Source**

The base class [PreTrainedModel] implements the common methods for loading/saving a model either from a local file or directory, or from a pretrained model configuration provided by the library (downloaded from HuggingFace's Hub). [PreTrainedModel] also implements a few methods which are common among all the models to:

**Target**

기본 클래스 [PreTrainedModel], [TFPreTrainedModel], [FlaxPreTrainedModel]는 로컬 파일과 디렉토리로부터 모델을 로드하고 저장하거나 또는 (허깅페이스 AWS S3 리포지토리로부터 다운로드된) 라이브러리에서 제공하는 사전 훈련된 모델 설정을 로드하고 저장하는 것을 지원하는 기본 메소드를 구현하였습니다. [PreTrainedModel]과 [TFPreTrainedModel]은 또한 모든 모델들을 공통적으로 지원하는 메소드 여러개를 구현하였습니다: 새 토큰이 단어장에 추가될 때, 입력 토큰 임베딩의 크기를 조정합니다.

### chunk 2
- source_chars: 210
- target_chars: 116
- length_ratio: 1.81

**Source**

resize the input token embeddings when new tokens are added to the vocabulary The other methods that are common to each model are defined in [~modelingutils.ModuleUtilsMixin] and [~generation.GenerationMixin].

**Target**

각 모델에 공통인 다른 메소드들은 다음의 클래스에서 정의됩니다. ~modelingutils.ModuleUtilsMixin 텍스트 생성을 위한 ~modelingtfutils.TFModuleUtilsMixin

## main_classes/peft.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/peft.md

### chunk 1
- source_chars: 333
- target_chars: 199
- length_ratio: 1.67

**Source**

The [~integrations.PeftAdapterMixin] provides functions from the PEFT library for managing adapters with Transformers. This mixin supports all non-prompt-learning PEFT methods (LoRA, IA3, AdaLoRA, and others). Prefix tuning methods (prompt tuning, prompt learning) aren't supported because they can't be injected into a torch module.

**Target**

[~integrations.PeftAdapterMixin]은 Transformers 라이브러리와 함께 어댑터를 관리할 수 있도록 PEFT 라이브러리의 함수들을 제공합니다. 이 믹스인은 현재 LoRA, IA3, AdaLora를 지원합니다. 프리픽스 튜닝 방법들(프롬프트 튜닝, 프롬프트 학습)은 torch 모듈에 삽입할 수 없는 구조이므로 지원되지 않습니다.

## main_classes/quantization.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/quantization.md

### chunk 1
- source_chars: 563
- target_chars: 325
- length_ratio: 1.73

**Source**

Quantization techniques reduce memory and computational costs by representing weights and activations with lower-precision data types like 8-bit integers (int8). This enables loading larger models you normally wouldn't be able to fit into memory, and speeding up inference. Transformers supports the AWQ and GPTQ quantization algorithms and it supports 8-bit and 4-bit quantization with bitsandbytes. Quantization techniques that aren't supported in Transformers can be added with the [HfQuantizer] class. Learn how to quantize models in the Quantization guide.

**Target**

양자화 기법은 가중치와 활성화를 8비트 정수(int8)와 같은 더 낮은 정밀도의 데이터 타입으로 표현함으로써 메모리와 계산 비용을 줄입니다. 이를 통해 일반적으로는 메모리에 올릴 수 없는 더 큰 모델을 로드할 수 있고, 추론 속도를 높일 수 있습니다. Transformers는 AWQ와 GPTQ 양자화 알고리즘을 지원하며, bitsandbytes를 통해 8비트와 4비트 양자화를 지원합니다. Transformers에서 지원되지 않는 양자화 기법들은 [HfQuantizer] 클래스를 통해 추가될 수 있습니다. 모델을 양자화하는 방법은 이 양자화 가이드를 통해 배울 수 있습니다.

## main_classes/text_generation.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/main_classes/text_generation.md

### chunk 1
- source_chars: 208
- target_chars: 165
- length_ratio: 1.26

**Source**

Each framework has a generate method for text generation implemented in their respective GenerationMixin class: PyTorch [~generation.GenerationMixin.generate] is implemented in [~generation.GenerationMixin].

**Target**

각 프레임워크에는 해당하는 GenerationMixin 클래스에서 구현된 텍스트 생성을 위한 generate 메소드가 있습니다: PyTorch에서는 [~generation.GenerationMixin.generate]가 [~generation.GenerationMixin]에 구현되어 있습니다.

### chunk 2
- source_chars: 423
- target_chars: 350
- length_ratio: 1.21

**Source**

You can parameterize the generate method with a [~generation.GenerationConfig] class instance. Please refer to this class for the complete list of generation parameters, which control the behavior of the generation method. To learn how to inspect a model's generation configuration, what are the defaults, how to change the parameters ad hoc, and how to create and save a customized generation configuration, refer to the

**Target**

TensorFlow에서는 [~generation.TFGenerationMixin.generate]가 [~generation.TFGenerationMixin]에 구현되어 있습니다. Flax/JAX에서는 [~generation.FlaxGenerationMixin.generate]가 [~generation.FlaxGenerationMixin]에 구현되어 있습니다. 사용하는 프레임워크에 상관없이, generate 메소드는 [~generation.GenerationConfig] 클래스 인스턴스로 매개변수화 할 수 있습니다. generate 메소드의 동작을 제어하는 모든 생성 매개변수 목록을 확인하려면 이 클래스를 참조하세요.

## model_doc/autoformer.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/autoformer.md

### chunk 1
- source_chars: 359
- target_chars: 195
- length_ratio: 1.84

**Source**

The Autoformer model was proposed in Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long. This model augments the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

**Target**

The Autoformer 모델은 Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long가 제안한 오토포머: 장기 시계열 예측을 위한 자기상관 분해 트랜스포머 라는 논문에서 소개 되었습니다. 이 모델은 트랜스포머를 심층 분해 아키텍처로 확장하여, 예측 과정에서 추세와 계절성 요소를 점진적으로 분해할 수 있습니다.

### chunk 2
- source_chars: 1567
- target_chars: 678
- length_ratio: 2.31

**Source**

The abstract from the paper is the following: Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of de...

**Target**

예측 시간을 연장하는 것은 극한 기상 조기 경보 및 장기 에너지 소비 계획과 같은 실제 응용 프로그램에 중요한 요구 사항입니다. 본 논문은 시계열의 장기 예측 문제를 연구합니다. 기존의 트랜스포머 기반 모델들은 장거리 종속성을 발견하기 위해 다양한 셀프 어텐션 메커니즘을 채택합니다. 그러나 장기 미래의 복잡한 시간적 패턴으로 인해 모델이 신뢰할 수 있는 종속성을 찾기 어렵습니다. 또한, 트랜스포머는 긴 시계열의 효율성을 위해 점별 셀프 어텐션의 희소 버전을 채택해야 하므로 정보 활용의 병목 현상이 발생합니다. 우리는 트랜스포머를 넘어서 자기상관 메커니즘을 갖춘 새로운 분해 아키텍처인 Autoformer를 설계했습니다. 우리는 시계열 분해의 전처리 관행을 깨고 이를 심층 모델의 기본 내부 블록으로 혁신했습니다. 이 설계는 Autoformer에 복잡한 시계열에 대한 점진적 분해 능력을 부여합니다. 또한, 확률 과정 이론에서 영감을 받아 시계열의 주기성을 기반으로 자기상관 메커니즘을 설계했으며, 이는 하위 시계열 수준에서 종속성 발견과 표현 집계를 수행합니다. 자기상관은 효율성과 정확도 면에서 셀프 어텐션를 능가합니다. 장기 예측에서 Autoformer는 에너지, 교통, 경제, 날씨, 질병 등 5가지 실용적 응용 분야를 포괄하는 6개 벤치마크에서 38%의 상대적 개선으로 최첨단 정확도를 달성했습니다.

## model_doc/code_llama.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/code_llama.md

### chunk 1
- source_chars: 700
- target_chars: 428
- length_ratio: 1.64

**Source**

Code Llama is a specialized family of large language models based on Llama 2 for coding tasks. It comes in different flavors - general code, Python-specific, and instruction-following variant - all available in 7B, 13B, 34B, and 70B parameters. Code Llama models can generate, explain, and even fill in missing parts of your code (called "infilling"). It can also handle very long contexts with stable generation up to 100k tokens, even though it was trained on sequences of 16K tokens. You can find all the original Code Llama checkpoints under the Code Llama collection. > Click on the Code Llama models in the right sidebar for more examples of how to apply Code Llama to different coding tasks.

**Target**

이 모델은 2023년 8월 24일에 공개되었으며, 2023년 8월 25일에 Hugging Face Transformers에 추가되었습니다. Code Llama는 코딩 작업에 특화된 대규모 언어 모델 계열로, Llama 2를 기반으로 개발되었습니다. 일반적인 코드, Python 특화, 명령어(지시) 기반 변형 등 다양한 버전으로 제공되며, 모두 7B, 13B, 34B, 70B 매개변수 크기로 사용할 수 있습니다. Code Llama 모델은 코드를 생성하고 설명하며, 코드의 누락된 부분을 채울 수도 있습니다. 이를 인필링(infilling)이라고 합니다. 16K 토큰 길이로 훈련되었지만, 최대 100K 토큰까지 안정적으로 생성하며 긴 컨텍스트도 처리할 수 있습니다. Code Llama 컬렉션에서 모든 원본 Code Llama 체크포인트를 찾을 수 있습니다.

### chunk 2
- source_chars: 375
- target_chars: 221
- length_ratio: 1.7

**Source**

The example below demonstrates how to generate code with [Pipeline], or the [AutoModel], and from the command line. Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the Quantization overview for more available quantization backends. The example below uses bitsandbytes to only quantize the weights to 4-bits.

**Target**

> 다양한 코딩 작업에 Code Llama를 적용하는 더 많은 예시를 보려면 오른쪽 사이드바의 Code Llama 모델을 클릭하세요. 아래 예시는 [Pipeline], [AutoModel], 그리고 명령줄에서 코드를 생성하는 방법을 보여줍니다. 양자화는 가중치를 더 낮은 정밀도로 표현하여 대규모 모델의 메모리 부담을 줄입니다. 더 많은 사용 가능한 양자화 백엔드는 양자화 개요를 참조하세요.

## model_doc/electra.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/electra.md

### chunk 1
- source_chars: 838
- target_chars: 1031
- length_ratio: 0.81

**Source**

ELECTRA modifies the pretraining objective of traditional masked language models like BERT. Instead of just masking tokens and asking the model to predict them, ELECTRA trains two models, a generator and a discriminator. The generator replaces some tokens with plausible alternatives and the discriminator (the model you'll actually use) learns to detect which tokens are original and which were replaced. This training approach is very efficient and scales to larger models while using considerably less compute. This approach is super efficient because ELECTRA learns from every single token in the input, not just the masked ones. That's why even the small ELECTRA models can match or outperform much larger models while using way less computing resources. You can find all the original ELECTRA checkpoints under the ELECTRA release.

**Target**

ELECTRA 모델은 ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators 논문에서 제안되었습니다. ELECTRA는 두가지 트랜스포머 모델인 생성 모델과 판별 모델을 학습시키는 새로운 사전학습 접근법입니다. 생성 모델의 역할은 시퀀스에 있는 토큰을 대체하는 것이며 마스킹된 언어 모델로 학습됩니다. 우리가 관심을 가진 판별 모델은 시퀀스에서 어떤 토큰이 생성 모델에 의해 대체되었는지 식별합니다. BERT와 같은 마스킹된 언어 모델(MLM) 사전학습 방법은 일부 토큰을 [MASK] 토큰으로 바꿔 손상시키고 난 뒤, 모델이 다시 원본 토큰을 복원하도록 학습합니다. 이런 방식은 다운스트림 NLP 작업을 전이할 때 좋은 성능을 내지만, 효과적으로 사용하기 위해서는 일반적으로 많은 양의 연산이 필요합니다. 따라서 대안으로, 대체 토큰 탐지라고 불리는 샘플-효과적인 사전학습을 제안합니다. 우리의 방법론은 입력에 마스킹을 하는 대신에 소형 생성 모델의 그럴듯한 대안 토큰으로 손상시킵니다. 그리고 나서, 모델이 손상된 토큰의 원래 토큰을 예측하도록 훈련시키는 대신, 판별 모델을 각각의 토큰이 생성 모델의 샘플로 손상되었는지 아닌지 학습합니다. 실험들은 통해 이 새로운 사전학습 방식은 마스킹된 일부 토큰에만 적용되는 기존 방식과 달리 모든 입력 토큰에 대해 학습이 이뤄지기 때문에 마스킹된 언어 모델(MLM)보다 더 효율적임을 입증하였습니다. 결과적으로 소개된 방식이 같은 모델 크기, 데이터, 연산량을 가진 BERT모델로 학습한 결과를 압도하는 문맥 표현 학습을 할 수 있다는 것을 확인했습니다. 특히 작은 모델에서 성능 향상이 두드러지며, 예를 들어 GPU 한 대로 4일간 학습한 모델이 30배 더 많은 계산 자원을 ...

### chunk 2
- source_chars: 268
- target_chars: 262
- length_ratio: 1.02

**Source**

> Click on the right sidebar for more examples of how to use ELECTRA for different language tasks like sequence classification, token classification, and question answering. The example below demonstrates how to classify text with [Pipeline] or the [AutoModel] class.

**Target**

이 모델은 lysandre이 기여했습니다. 원본 코드는 이곳에서 찾아보실 수 있습니다. ELECTRA는 사전학습 방법으로 기본 모델인 BERT의 구조와 거의 차이가 없습니다. 유일한 차이는 임베딩 크기와 히든 크기를 구분했다는 점입니다. 임베딩 크기는 일반적으로 더 작고, 히든 크기는 더 큽니다. 임베딩에서 임베딩 크기를 히든 크기로 변환하기 위해 추가로 선형 변환 층이 사용됩니다. 임베딩 크기와 히든 크기가 동일할 경우에는 이 선형 변환 층이 필요하지 않습니다.

## model_doc/gemma3.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/gemma3.md

### chunk 1
- source_chars: 715
- target_chars: 391
- length_ratio: 1.83

**Source**

Gemma 3 is a multimodal model with pretrained and instruction-tuned variants, available in 1B, 13B, and 27B parameters. The architecture is mostly the same as the previous Gemma versions. The key differences are alternating 5 local sliding window self-attention layers for every global self-attention layer, support for a longer context length of 128K tokens, and a SigLip encoder that can "pan & scan" high-resolution images to prevent information from disappearing in high resolution images or images with non-square aspect ratios. The instruction-tuned variant was post-trained with knowledge distillation and reinforcement learning. You can find all the original Gemma 3 checkpoints under the Gemma 3 release.

**Target**

Gemma 3는 사전 훈련된 버전과 지시문 조정 버전을 갖춘 멀티모달 모델로, 1B, 13B, 27B 매개변수로 제공됩니다. 아키텍처는 이전 Gemma 버전과 대부분 동일합니다. 주요 차이점은 모든 글로벌 셀프 어텐션 레이어마다 5개의 로컬 슬라이딩 윈도우 셀프 어텐션 레이어를 번갈아 사용하는 점, 128K 토큰의 더 긴 컨텍스트 길이를 지원하는 점, 그리고 고해상도 이미지나 정사각형이 아닌 종횡비의 이미지에서 정보가 사라지는 것을 방지하기 위해 고해상도 이미지를 "패닝 및 스캐닝"할 수 있는 SigLip 인코더를 사용한다는 점입니다. 지시문 조정 버전은 지식 증류 및 강화 학습으로 후속 학습되었습니다. Gemma 3의 모든 원본 체크포인트는 Gemma 3 릴리스에서 확인할 수 있습니다.

### chunk 2
- source_chars: 426
- target_chars: 236
- length_ratio: 1.81

**Source**

> Click on the Gemma 3 models in the right sidebar for more examples of how to apply Gemma to different vision and language tasks. The example below demonstrates how to generate text based on an image with [Pipeline] or the [AutoModel] class. Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the Quantization overview for more available quantization backends.

**Target**

> Gemma를 다양한 비전 및 언어 작업에 적용하는 추가 예시를 보려면 오른쪽 사이드바의 Gemma 3 모델을 클릭하세요. 아래 예시는 [Pipeline] 또는 [AutoModel] 클래스를 사용하여 이미지를 기반으로 텍스트를 생성하는 방법을 보여줍니다. 양자화는 가중치를 더 낮은 정밀도로 표현하여, 큰 모델의 메모리 부담을 줄여줍니다. 사용 가능한 양자화 백엔드에 대한 더 자세한 내용은 양자화 개요를 참고하세요.

## model_doc/gpt2.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/gpt2.md

### chunk 1
- source_chars: 651
- target_chars: 329
- length_ratio: 1.98

**Source**

GPT-2 is a scaled up version of GPT, a causal transformer language model, with 10x more parameters and training data. The model was pretrained on a 40GB dataset to predict the next word in a sequence based on all the previous words. This approach enabled the model to perform many downstream tasks in a zero-shot setting. The blog post released by OpenAI can be found here. The model architecture uses a unidirectional (causal) attention mechanism where each token can only attend to previous tokens, making it particularly effective for text generation tasks. You can find all the original GPT-2 checkpoints under the OpenAI community organization.

**Target**

GPT-2는 GPT의 확장 버전으로, 인과적 트랜스포머 언어 모델이며, 10배 더 많은 매개변수와 학습 데이터를 가지고 있습니다. 이 모델은 이전의 모든 단어를 기반으로 다음 단어를 예측하도록 40GB 데이터 세트에서 사전 학습되었습니다. 이러한 접근 방식을 통해 이 모델은 제로샷 설정에서 많은 다운스트림 작업을 수행할 수 있게 되었습니다. 모델 아키텍처는 각 토큰이 이전 토큰에만 주의를 기울일 수 있는 단방향(인과적) 어텐션 메커니즘을 사용하므로, 텍스트 생성 작업에 특히 효과적입니다. 모든 원본 GPT-2 체크포인트는 OpenAI community 조직에서 찾을 수 있습니다.

### chunk 2
- source_chars: 305
- target_chars: 190
- length_ratio: 1.61

**Source**

> Click on the GPT-2 models in the right sidebar for more examples of how to apply GPT-2 to different language tasks. The example below demonstrates how to generate text with [Pipeline] or the [AutoModel], and from the command line. One can also serve the model using vLLM with the transformers backend.

**Target**

> 오른쪽 사이드바의 GPT-2 모델을 클릭하여 GPT-2를 다양한 언어 작업에 적용하는 더 많은 예시를 확인하세요. 아래 예시는 [Pipeline] 또는 [AutoModel], 그리고 명령줄에서 GPT-2로 텍스트를 생성하는 방법을 보여줍니다. transformers backend를 사용하여 vLLM으로 모델을 서빙할 수도 있습니다.

## model_doc/jamba.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/jamba.md

### chunk 1
- source_chars: 790
- target_chars: 459
- length_ratio: 1.72

**Source**

Jamba is a hybrid Transformer-Mamba mixture-of-experts (MoE) language model ranging from 52B to 398B total parameters. This model aims to combine the advantages of both model families, the performance of transformer models and the efficiency and longer context (256K tokens) of state space models (SSMs) like Mamba. Jamba's architecture features a blocks-and-layers approach that allows Jamba to successfully integrate Transformer and Mamba architectures altogether. Each Jamba block contains either an attention or a Mamba layer, followed by a multi-layer perceptron (MLP), producing an overall ratio of one Transformer layer out of every eight total layers. MoE layers are mixed in to increase model capacity. You can find all the original Jamba checkpoints under the AI21 organization.

**Target**

Jamba는 Transformer와 Mamba 기반의 하이브리드 전문가 혼합(MoE) 언어 모델로, 총 매개변수 수는 52B에서 398B까지 다양합니다. 이 모델은 Transformer 모델의 성능과 Mamba와 같은 상태 공간 모델의 효율성 및 긴 컨텍스트 처리 능력(256K 토큰)을 모두 활용하는 것을 목표로 합니다. Jamba의 아키텍처는 블록과 레이어 기반 구조를 사용하여 Transformer와 Mamba 아키텍처를 통합할 수 있도록 설계되었습니다. 각 Jamba 블록은 어텐션 레이어 또는 Mamba 레이어 중 하나와 그 뒤를 잇는 다층 퍼셉트론(MLP)으로 구성되어 있습니다. Transformer 레이어는 8개의 레이어 중 하나의 비율로 주기적으로 배치됩니다. 또한 모델 용량을 확장하기 위해 MoE 레이어가 혼합되어 있습니다. 모든 원본 Jamba 체크포인트는 AI21 조직에서 확인할 수 있습니다.

### chunk 2
- source_chars: 410
- target_chars: 234
- length_ratio: 1.75

**Source**

> Click on the Jamba models in the right sidebar for more examples of how to apply Jamba to different language tasks. The example below demonstrates how to generate text with [Pipeline], [AutoModel], and from the command line. Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the Quantization overview for more available quantization backends.

**Target**

> 오른쪽 사이드바에 있는 Jamba 모델을 누르면 다양한 언어 작업에 Jamba를 적용하는 예제를 더 확인할 수 있습니다. 아래 예제는 [Pipeline]과 [AutoModel], 그리고 커맨드라인을 통해 텍스트를 생성하는 방법을 보여줍니다. 양자화는 가중치를 더 낮은 정밀도로 표현하여 대규모 모델의 메모리 부담을 줄여줍니다. 사용할 수 있는 다양한 양자화 백엔드에 대해서는 Quantization를 참고하세요.

## model_doc/lfm2.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/lfm2.md

### chunk 1
- source_chars: 402
- target_chars: 314
- length_ratio: 1.28

**Source**

LFM2 represents a new generation of Liquid Foundation Models developed by Liquid AI, specifically designed for edge AI and on-device deployment. The models are available in four sizes (350M, 700M, 1.2B, and 2.6B parameters) and are engineered to run efficiently on CPU, GPU, and NPU hardware, making them particularly well-suited for applications requiring low latency, offline operation, and privacy.

**Target**

이 모델은 2025년 7월 10일에 출시되었으며, 2025년 7월 10일에 Hugging Face Transformers에 추가되었습니다. LFM2는 Liquid AI가 개발한 차세대 Liquid Foundation Model로 egde AI와 온디바이스 배포에 특화되어 설계되었습니다. 이 모델들은 350M, 700M, 1.2B, 2.6B의 네 가지 크기의 매개변수로 제공되며, CPU, GPU, NPU 하드웨어에서 효율적으로 실행되도록 설계되었습니다. 이로 인해 특히 낮은 지연 시간, 오프라인 작동 및 개인 정보 보호가 필요한 애플리케이션에 적합합니다.

### chunk 2
- source_chars: 1064
- target_chars: 520
- length_ratio: 2.05

**Source**

The architecture consists of blocks of gated short convolution blocks and blocks of grouped query attention with QK layernorm. This design stems from the concept of dynamical systems, where linear operations are modulated by input-dependent gates. The short convolutions are particularly optimized for embedded SoC CPUs, making them ideal for devices that require fast, local inference without relying on cloud connectivity. LFM2 was designed to maximize quality under strict speed and memory constraints. This was accomplished through a systematic architecture search to optimize the models for real-world performance on embedded hardware by measuring actual peak memory usage and inference speed on Qualcomm Snapdragon processors. This results in models that achieve 2x faster decode and prefill performance compared to similar-sized models, while maintaining superior benchmark performance acro...

**Target**

아키텍처는 게이트가 있는 짧은 합성곱 블록과 QK 레이어 정규화가 적용된 그룹 쿼리 어텐션 블록으로 구성됩니다. 이 설계는 선형 연산이 입력 의존적인 게이트에 의해 조절되는 동적 시스템 개념에서 비롯되었습니다. 짧은 합성곱은 특히 임베디드 SoC CPU에 최적화되어 있어, 클라우드 연결에 의존하지 않고 빠르고 로컬화된 추론이 필요한 장치에 이상적입니다. LFM2는 제한된 속도와 메모리 환경에서 품질을 최대화되도록 설계되었습니다. 이는 퀄컴 스냅드래곤 프로세서에서 실제 최대 메모리 사용량과 추론 속도를 측정하여, 임베디드 하드웨어에서의 실제 성능에 맞게 모델을 최적화하기 위한 체계적인 아키텍처 탐색을 통해 달성되었습니다. 그 결과, 비슷한 크기의 모델에 비해 2배 빠른 디코딩 및 프리필 성능을 달성하면서도, 지식, 수학, 지시 사항 따르기, 다국어 작업 전반에서 우수한 벤치마크 성능을 유지하는 모델이 탄생했습니다. 다음 예시는 AutoModelForCausalLM 클래스를 사용하여 답변을 생성하는 방법을 보여줍니다.

## model_sharing.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_sharing.md

### chunk 1
- source_chars: 492
- target_chars: 358
- length_ratio: 1.37

**Source**

The Hugging Face Hub is a platform for sharing, discovering, and consuming models of all different types and sizes. We highly recommend sharing your model on the Hub to push open-source machine learning forward for everyone! This guide will show you how to share a model to the Hub from Transformers. To share a model to the Hub, you need a Hugging Face account. Create a User Access Token (stored in the cache by default) and login to your account from either the command line or notebook.

**Target**

지난 두 튜토리얼에서 분산 설정을 위해 PyTorch, Keras 및 🤗 Accelerate를 사용하여 모델을 미세 조정하는 방법을 보았습니다. 다음 단계는 모델을 커뮤니티와 공유하는 것입니다! Hugging Face는 인공지능의 민주화를 위해 모두에게 지식과 자원을 공개적으로 공유해야 한다고 믿습니다. 다른 사람들이 시간과 자원을 절약할 수 있도록 커뮤니티에 모델을 공유하는 것을 고려해 보세요. 이 튜토리얼에서 Model Hub에서 훈련되거나 미세 조정 모델을 공유하는 두 가지 방법에 대해 알아봅시다: 커뮤니티에 모델을 공유하려면, huggingface.co에 계정이 필요합니다. 기존 조직에 가입하거나 새로 만들 수도 있습니다.

### chunk 2
- source_chars: 369
- target_chars: 274
- length_ratio: 1.35

**Source**

Each model repository features versioning, commit history, and diff visualization. Versioning is based on Git and Git Large File Storage (LFS), and it enables revisions, a way to specify a model version with a commit hash, tag or branch. For example, use the revision parameter in [~PreTrainedModel.frompretrained] to load a specific model version from a commit hash.

**Target**

모델 허브의 각 저장소는 일반적인 GitHub 저장소처럼 작동합니다. 저장소는 버전 관리, 커밋 기록, 차이점 시각화 기능을 제공합니다. 모델 허브에 내장된 버전 관리는 git 및 git-lfs를 기반으로 합니다. 즉, 하나의 모델을 하나의 저장소로 취급하여 접근 제어 및 확장성이 향상됩니다. 버전 제어는 커밋 해시, 태그 또는 브랜치로 모델의 특정 버전을 고정하는 방법인 revision을 허용합니다. 따라서 revision 매개변수를 사용하여 특정 모델 버전을 가져올 수 있습니다:

## models.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/models.md

### chunk 1
- source_chars: 581
- target_chars: 375
- length_ratio: 1.55

**Source**

Transformers provides many pretrained models that are ready to use with a single line of code. It requires a model class and the [~PreTrainedModel.frompretrained] method. Call [~PreTrainedModel.frompretrained] to download and load a model's weights and configuration stored on the Hugging Face Hub. > The [~PreTrainedModel.frompretrained] method loads weights stored in the safetensors file format if they're available. Traditionally, PyTorch model weights are serialized with the pickle utility which is known to be unsecure. Safetensor files are more secure and faster to load.

**Target**

Transformers는 한 줄의 코드로 사용할 수 있는 많은 사전 훈련된 모델을 제공합니다. 모델 클래스와 [~PreTrainedModel.frompretrained] 메소드가 필요합니다. [~PreTrainedModel.frompretrained]를 호출하여 Hugging Face Hub에 저장된 모델의 가중치와 구성을 다운로드하고 로드하세요. > [~PreTrainedModel.frompretrained] 메소드는 safetensors 파일 형식으로 저장된 가중치가 있으면 이를 로드합니다. 전통적으로 PyTorch 모델 가중치는 보안에 취약한 것으로 알려진 pickle 유틸리티로 직렬화됩니다. Safetensor 파일은 더 안전하고 로드 속도가 빠릅니다.

### chunk 2
- source_chars: 938
- target_chars: 599
- length_ratio: 1.57

**Source**

This guide explains how models are loaded, the different ways you can load a model, how to overcome memory issues for really big models, and how to load custom models. All models have a configuration.py file with specific attributes like the number of hidden layers, vocabulary size, activation function, and more. You'll also find a modeling.py file that defines the layers and mathematical operations taking place inside each layer. The modeling.py file takes the model attributes in configuration.py and builds the model accordingly. At this point, you have a model with random weights that needs to be trained to output meaningful results. > An architecture refers to the model's skeleton and a checkpoint refers to the model's weights for a given architecture. For example, BERT is an architecture while google-bert/bert-base-uncased is a checkpoint. You'll see the term model used interchang...

**Target**

이 가이드는 모델을 불러오는 방법, 다양한 로딩 방식, 매우 큰 모델에서 발생할 수 있는 메모리 문제를 해결하는 방법, 그리고 사용자 정의 모델을 불러오는 방법을 설명합니다. 모든 모델에는 은닉 레이어 수, 어휘 사전 크기, 활성화 함수 등과 같은 특정 속성이 포함된 configuration.py 파일이 있습니다. 또한 각 레이어의 정의와 각각의 레이어 안에서 일어나는 수학적 연산을 정의하는 modeling.py 파일도 있습니다. modeling.py 파일은 configuration.py에 정의된 모델 속성을 바탕으로 모델을 구축합니다. 이 단계에서는 아직 학습되지 않은 무작위 가중치를 가진 상태이기 때문에, 의미 있는 출력을 얻기 위해서는 학습이 필요합니다. > 아키텍처(Architecture)는 모델의 골격을 의미하고 체크포인트(checkpoint)는 주어진 아키텍처에 대한 모델의 가중치를 의미합니다. 예를 들어, BERT는 아키텍처이고 google-bert/bert-base-uncased는 해당 아키텍처의 체크포인트(checkpoint)입니다. 모델이라는 용어는 아키텍처 및 체크포인트(checkpoint)와 혼용하여 사용되는 것을 볼 수 있습니다.

## pipeline_gradio.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/pipeline_gradio.md

### chunk 1
- source_chars: 425
- target_chars: 259
- length_ratio: 1.64

**Source**

Gradio, a fast and easy library for building and sharing machine learning apps, is integrated with [Pipeline] to quickly create a simple interface for inference. Before you begin, make sure Gradio is installed. Create a pipeline for your task, and then pass it to Gradio's Interface.frompipeline function to create the interface. Gradio automatically determines the appropriate input and output components for a [Pipeline].

**Target**

머신러닝 앱을 빠르고 쉽게 구축하고 공유할 수 있는 라이브러리인 Gradio는 [Pipeline]과 통합되어 추론을 위한 간단한 인터페이스를 빠르게 생성할 수 있습니다. 시작하기 전에 Gradio가 설치되어 있는지 확인하세요. 원하는 작업에 맞는 pipeline을 생성한 다음, Gradio의 Interface.frompipeline 함수에 전달하여 인터페이스를 만드세요. Gradio는 [Pipeline]에 맞는 입력 및 출력 컴포넌트를 자동으로 결정합니다.

### chunk 2
- source_chars: 338
- target_chars: 217
- length_ratio: 1.56

**Source**

Add launch to create a web server and start up the app. The web app runs on a local server by default. To share the app with other users, set share=True in launch to generate a temporary public link. For a more permanent solution, host the app on Hugging Face Spaces. The Space below is created with the code above and hosted on Spaces.

**Target**

launch를 추가하여 웹 서버를 생성하고 앱을 시작하세요. 웹 앱은 기본적으로 로컬 서버에서 실행됩니다. 다른 사용자와 앱을 공유하려면 launch에서 share=True로 설정하여 임시 공개 링크를 생성하세요. 더 지속적인 솔루션을 원한다면 Hugging Face Spaces에서 앱을 호스팅하세요. 아래 Space는 위 코드를 사용하여 생성되었으며, Spaces에서 호스팅됩니다.

## quantization/quanto.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/quantization/quanto.md

### chunk 1
- source_chars: 282
- target_chars: 174
- length_ratio: 1.62

**Source**

Quanto is a PyTorch quantization backend for Optimum. It features linear quantization for weights (float8, int8, int4, int2) with accuracy very similar to full-precision models. Quanto is compatible with any model modality and device, making it simple to use regardless of hardware.

**Target**

이 노트북으로 Quanto와 transformers를 사용해 보세요! 🤗 Quanto 라이브러리는 다목적 파이토치 양자화 툴킷입니다. 이 라이브러리에서 사용되는 양자화 방법은 선형 양자화입니다. Quanto는 다음과 같은 여러 가지 기능을 제공합니다: 가중치 양자화 (float8,int8,int4,int2)

### chunk 3
- source_chars: 387
- target_chars: 366
- length_ratio: 1.06

**Source**

Install Quanto with the following command. Quantize a model by creating a [QuantoConfig] and specifying the weights parameter to quantize to. This works for any model in any modality as long as it contains torch.nn.Linear layers. > The Transformers integration only supports weight quantization. Use the Quanto library directly if you need activation quantization, calibration, or QAT.

**Target**

이제 [~PreTrainedModel.frompretrained] 메소드에 [QuantoConfig] 객체를 전달하여 모델을 양자화할 수 있습니다. 이 방식은 torch.nn.Linear 레이어를 포함하는 모든 모달리티의 모든 모델에서 잘 작동합니다. 허깅페이스의 transformers 라이브러리는 개발자 편의를 위해 quanto의 인터페이스를 일부 통합하여 지원하고 있으며, 이 방식으로는 가중치 양자화만 지원합니다. 활성화 양자화, 캘리브레이션, QAT 같은 더 복잡한 기능을 수행하기 위해서는 quanto 라이브러리의 해당 함수를 직접 호출해야 합니다. 참고로, transformers에서는 아직 직렬화가 지원되지 않지만 곧 지원될 예정입니다!
