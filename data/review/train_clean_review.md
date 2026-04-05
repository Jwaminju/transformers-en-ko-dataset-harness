# Train Alignment Review

- documents: 37
- chunks: 66

## fsdp.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/fsdp.md

### chunk 1
- source_chars: 557
- target_chars: 345
- length_ratio: 1.61

**Source**

Fully Sharded Data Parallel (FSDP) is a parallelism method that combines the advantages of data and model parallelism for distributed training. Unlike DistributedDataParallel (DDP), FSDP saves more memory because it doesn't replicate a model on each GPU. It shards the models parameters, gradients and optimizer states across GPUs. Each model shard processes a portion of the data and the results are synchronized to speed up training. This guide covers how to set up training a model with FSDP and Accelerate, a library for managing distributed training.

**Target**

Fully Sharded Data Parallel (FSDP)은 모델의 매개변수, 그레이디언트 및 옵티마이저 상태를 사용 가능한 GPU(작업자 또는 랭크라고도 함) 수에 따라 분할하는 데이터 병렬 처리 방식입니다. DistributedDataParallel (DDP)와 달리, FSDP는 각 GPU에 모델을 복제하기 때문에 메모리 사용량을 줄입니다. 이는 GPU 메모리 효율성을 향상시키며 적은 수의 GPU로 훨씬 더 큰 모델을 훈련할 수 있게 합니다. FSDP는 분산 환경에서의 훈련을 쉽게 관리할 수 있는 라이브러리인 Accelerate와 통합되어 있으며, 따라서 [Trainer] 클래스에서 사용할 수 있습니다.

### chunk 2
- source_chars: 507
- target_chars: 357
- length_ratio: 1.42

**Source**

Always start by running the accelerate config command to help Accelerate set up the correct distributed training environment. The section below discusses some of the more important FSDP configuration options. Learn more about other available options in the fsdpconfig parameter. FSDP offers several sharding strategies to distribute a model. Refer to the table below to help you choose the best strategy for your setup. Specify a strategy with the fsdpshardingstrategy parameter in the configuration file.

**Target**

시작하기 전에 Accelerate가 설치되어 있고 최소 PyTorch 2.1.0 이상의 버전이 설치되어 있는지 확인하세요. 시작하려면 accelerate config 명령을 실행하여 훈련 환경에 대한 구성 파일을 생성하세요. Accelerate는 이 구성 파일을 사용하여 accelerate config에서 선택한 훈련 옵션에 따라 자동으로 올바른 훈련 환경을 설정합니다. accelerate config를 실행하면 훈련 환경을 구성하기 위한 일련의 옵션들이 나타납니다. 이 섹션에서는 가장 중요한 FSDP 옵션 중 일부를 다룹니다. 다른 사용 가능한 FSDP 옵션에 대해 더 알아보고 싶다면 fsdpconfig 매개변수를 참조하세요.

## gguf.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/gguf.md

### chunk 1
- source_chars: 204
- target_chars: 177
- length_ratio: 1.15

**Source**

GGUF is a file format used to store models for inference with GGML, a fast and lightweight inference framework written in C and C++. GGUF is a single-file format containing the model metadata and tensors.

**Target**

GGUF 파일 형식은 GGML과 그에 의존하는 다른 라이브러리, 예를 들어 매우 인기 있는 llama.cpp이나 whisper.cpp에서 추론을 위한 모델을 저장하는데 사용됩니다. 이 파일 형식은 Hugging Face Hub에서 지원되며, 파일 내의 텐서와 메타데이터를 신속하게 검사할 수 있는 기능을 제공합니다.

### chunk 2
- source_chars: 280
- target_chars: 434
- length_ratio: 0.65

**Source**

The GGUF format also supports many quantized data types (refer to quantization type table for a complete list of supported quantization types) which saves a significant amount of memory, making inference with large models like Whisper and Llama feasible on local and edge devices.

**Target**

이 형식은 "단일 파일 형식(single-file-format)"으로 설계되었으며, 하나의 파일에 설정 속성, 토크나이저 어휘, 기타 속성뿐만 아니라 모델에서 로드되는 모든 텐서가 포함됩니다. 이 파일들은 파일의 양자화 유형에 따라 다른 형식으로 제공됩니다. 다양한 양자화 유형에 대한 간략한 설명은 여기에서 확인할 수 있습니다. transformers 내에서 gguf 파일을 로드할 수 있는 기능을 추가하여 GGUF 모델의 추가 학습/미세 조정을 제공한 후 ggml 생태계에서 다시 사용할 수 있도록 gguf 파일로 변환하는 기능을 제공합니다. 모델을 로드할 때 먼저 FP32로 역양자화한 후, PyTorch에서 사용할 수 있도록 가중치를 로드합니다. > 지원은 아직 초기 단계에 있으며, 다양한 양자화 유형과 모델 아키텍처에 대해 이를 강화하기 위한 기여를 환영합니다.

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

## model_doc/albert.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/albert.md

### chunk 1
- source_chars: 851
- target_chars: 439
- length_ratio: 1.94

**Source**

ALBERT is designed to address memory limitations of scaling and training of BERT. It adds two parameter reduction techniques. The first, factorized embedding parametrization, splits the larger vocabulary embedding matrix into two smaller matrices so you can grow the hidden size without adding a lot more parameters. The second, cross-layer parameter sharing, allows layer to share parameters which keeps the number of learnable parameters lower. ALBERT was created to address problems like -- GPU/TPU memory limitations, longer training times, and unexpected model degradation in BERT. ALBERT uses two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT: Factorized embedding parameterization: The large vocabulary embedding matrix is decomposed into two smaller matrices, reducing memory consumption.

**Target**

ALBERT는 BERT의 확장성과 학습 시 메모리 한계를 해결하기 위해 설계된 모델입니다. 이 모델은 두 가지 파라미터 감소 기법을 도입합니다. 첫 번째는 임베딩 행렬 분해(factorized embedding parametrization)로, 큰 어휘 임베딩 행렬을 두 개의 작은 행렬로 분해하여 히든 사이즈를 늘려도 파라미터 수가 크게 증가하지 않도록 합니다. 두 번째는 계층 간 파라미터 공유(cross-layer parameter sharing)로, 여러 계층이 파라미터를 공유하여 학습해야 할 파라미터 수를 줄입니다. ALBERT는 BERT에서 발생하는 GPU/TPU 메모리 한계, 긴 학습 시간, 갑작스런 성능 저하 문제를 해결하기 위해 만들어졌습니다. ALBERT는 파라미터를 줄이기 위해 두 가지 기법을 사용하여 메모리 사용량을 줄이고 BERT의 학습 속도를 높입니다:

### chunk 2
- source_chars: 462
- target_chars: 307
- length_ratio: 1.5

**Source**

Cross-layer parameter sharing: Instead of learning separate parameters for each transformer layer, ALBERT shares parameters across layers, further reducing the number of learnable weights. ALBERT uses absolute position embeddings (like BERT) so padding is applied at right. Size of embeddings is 128 While BERT uses 768. ALBERT can processes maximum 512 token at a time. You can find all the original ALBERT checkpoints under the ALBERT community organization.

**Target**

임베딩 행렬 분해: 큰 어휘 임베딩 행렬을 두 개의 더 작은 행렬로 분해하여 메모리 사용량을 줄입니다. 계층 간 파라미터 공유: 각 트랜스포머 계층마다 별도의 파라미터를 학습하는 대신, 여러 계층이 파라미터를 공유하여 학습해야 할 가중치 수를 더욱 줄입니다. ALBERT는 BERT와 마찬가지로 절대 위치 임베딩(absolute position embeddings)을 사용하므로, 입력 패딩은 오른쪽에 적용해야 합니다. 임베딩 크기는 128이며, BERT의 768보다 작습니다. ALBERT는 한 번에 최대 512개의 토큰을 처리할 수 있습니다.

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

## model_doc/llama3.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/llama3.md

### chunk 1
- source_chars: 1015
- target_chars: 661
- length_ratio: 1.54

**Source**

The Llama3 model was proposed in Introducing Meta Llama 3: The most capable openly available LLM to date by the meta AI team. The abstract from the blogpost is the following: Today, we’re excited to share the first two models of the next generation of Llama, Meta Llama 3, available for broad use. This release features pretrained and instruction-fine-tuned language models with 8B and 70B parameters that can support a broad range of use cases. This next generation of Llama demonstrates state-of-the-art performance on a wide range of industry benchmarks and offers new capabilities, including improved reasoning. We believe these are the best open source models of their class, period. In support of our longstanding open approach, we’re putting Llama 3 in the hands of the community. We want to kickstart the next wave of innovation in AI across the stack—from applications to developer tools ...

**Target**

라마3 모델은 Meta AI 팀이 제안한 메타 라마3 소개: 현재까지 가장 유능한 공개 가능 LLM에서 소개되었습니다. 오늘, 광범위한 사용을 위해 이용 가능한 라마의 차세대 모델인 메타 라마3의 첫 두 모델을 공유하게 되어 기쁩니다. 이번 출시는 8B와 70B 매개변수를 가진 사전 훈련 및 지시 미세 조정된 언어 모델을 특징으로 하며, 광범위한 사용 사례를 지원할 수 있습니다. 라마의 이 차세대 모델은 다양한 산업 벤치마크에서 최첨단의 성능을 보여주며, 개선된 추론 능력을 포함한 새로운 기능을 제공합니다. 우리는 이것들이 단연코 해당 클래스에서 최고의 오픈 소스 모델이라고 믿습니다. 오랜 개방적 접근 방식을 지지하며, 우리는 라마3를 커뮤니티 기여자들에게 맡기고 있습니다. 애플리케이션에서 개발자 도구, 평가, 추론 최적화 등에 이르기까지 AI 스택 전반에 걸친 다음 혁신의 물결을 촉발하길 희망합니다. 여러분이 무엇을 만들지 기대하며 여러분의 피드백을 고대합니다. 라마3 모델들은 bfloat16를 사용하여 훈련되었지만, 원래의 추론은 float16을 사용합니다. Hub에 업로드된 체크포인트들은 dtype = 'float16'을 사용하는데, 이는 AutoModel API가 체크포인트를 torch.float32에서 torch.float16으로 변환하는데 이용됩니다.

### chunk 2
- source_chars: 258
- target_chars: 268
- length_ratio: 0.96

**Source**

Checkout all Llama3 model checkpoints here. The original code of the authors can be found here. The Llama3 models were trained using bfloat16, but the original inference uses float16. The checkpoints uploaded on the Hub use dtype = 'float16', which will be

**Target**

model = AutoModelForCausalLM.frompretrained("path", dtype = "auto")를 사용하여 모델을 초기화할 때, 온라인 가중치의 dtype는 dtype="auto"를 사용하지 않는 한 대부분 무관합니다. 그 이유는 모델이 먼저 다운로드되고(온라인 체크포인트의 dtype를 사용), 그 다음 torch의 dtype으로 변환되어(torch.float32가 됨), 마지막으로 config에 dtype이 제공된 경우 가중치가 사용되기 때문입니다.

## model_doc/mamba.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/mamba.md

### chunk 1
- source_chars: 722
- target_chars: 998
- length_ratio: 0.72

**Source**

Mamba is a selective structured state space model (SSMs) designed to work around Transformers computational inefficiency when dealing with long sequences. It is a completely attention-free architecture, and comprised of a combination of H3 and gated MLP blocks (Mamba block). Mamba's "content-based reasoning" allows it to focus on specific parts of an input depending on the current token. Mamba also uses a new hardware-aware parallel algorithm to compensate for the lack of convolutional operations. As a result, Mamba has fast inference and can scale to very long sequences. You can find all the original Mamba checkpoints under the State Space Models organization. > This model was contributed by Molbap and AntonV.

**Target**

맘바(Mamba) 모델은 Albert Gu, Tri Dao가 제안한 맘바: 선택적 상태 공간을 이용한 선형 시간 시퀀스 모델링라는 논문에서 소개 되었습니다. 이 모델은 state-space-models을 기반으로 한 새로운 패러다임 아키텍처입니다. 직관적인 이해를 얻고 싶다면 이곳을 참고 하세요. 현재 딥러닝에서 흥미로운 응용 프로그램을 구동하는 대부분의 기초 모델들은 거의 보편적으로 트랜스포머 아키텍처와 그 핵심 어텐션 모듈을 기반으로 합니다. 선형 어텐션, 게이트된 컨볼루션과 순환 모델, 구조화된 상태 공간 모델(SSM) 등 많은 준이차시간(subquadratic-time) 아키텍처가 긴 시퀀스에 대한 트랜스포머의 계산 비효율성을 해결하기 위해 개발되었지만, 언어와 같은 중요한 양식에서는 어텐션만큼 성능을 내지 못했습니다. 우리는 이러한 모델의 주요 약점이 내용 기반 추론을 수행하지 못한다는 점임을 알고 몇 가지를 개선했습니다. 첫째, SSM 매개변수를 입력의 함수로 만드는 것만으로도 이산 모달리티(discrete modalities)의 약점을 해결할 수 있어, 현재 토큰에 따라 시퀀스 길이 차원을 따라 정보를 선택적으로 전파하거나 잊을 수 있게 합니다. 둘째, 이러한 변경으로 효율적인 컨볼루션을 사용할 수 없게 되었지만, 우리는 순환 모드에서 하드웨어를 인식하는 병렬 알고리즘을 설계했습니다. 우리는 이러한 선택적 SSM을 어텐션이나 MLP 블록도 없는 단순화된 종단간 신경망 아키텍처인 맘바에 통합시켰습니다. 맘바는 빠른 추론(트랜스포머보다 5배 높은 처리량)과 시퀀스 길이에 대한 선형 확장성을 누리며, 백만 길이 시퀀스까지 실제 데이터에서 성능이 향상됩니다. 일반적인 시퀀스 모델 백본으로서 맘바는 언어, 오디오, 유전체학과 같은 여러 양식에서 최첨단 성능을 달성합니다. 언어...

### chunk 2
- source_chars: 410
- target_chars: 192
- length_ratio: 2.14

**Source**

> Click on the Mamba models in the right sidebar for more examples of how to apply Mamba to different language tasks. The example below demonstrates how to generate text with [Pipeline], [AutoModel], and from the command line. Quantization reduces the memory burden of large models by representing the weights in a lower precision. Refer to the Quantization overview for more available quantization backends.

**Target**

맘바는 고전적인 트랜스포머와 견줄 만한 새로운 상태 공간 모델 아키텍처입니다. 이는 구조화된 상태 공간 모델의 발전 선상에 있으며, 플래시어텐션의 정신을 따르는 효율적인 하드웨어 인식 설계와 구현을 특징으로 합니다. 맘바는 어텐션 레이어와 동등한 믹서(mixer) 레이어를 쌓습니다. 맘바의 핵심 로직은 MambaMixer 클래스에 있습니다.

## model_doc/smolvlm.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/smolvlm.md

### chunk 1
- source_chars: 165
- target_chars: 168
- length_ratio: 0.98

**Source**

SmolVLM2 (blog post) is an adaptation of the Idefics3 model with two main differences: It uses SmolLM2 for the text model. It supports multi-image and video inputs

**Target**

이 모델은 2025년 2월 20일에 출시되었으며, 동시에 허깅페이스 Transformer 라이브러리에 추가되었습니다. SmolVLM2 (블로그 글) 은 Idefics3 모델을 개선한 버전으로, 두 가지 주요 차이점이 있습니다: 한 장의 이미지뿐 아니라 여러 장의 이미지와 비디오 입력도 지원합니다.

### chunk 2
- source_chars: 312
- target_chars: 289
- length_ratio: 1.08

**Source**

Input images are processed either by upsampling (if resizing is enabled) or at their original resolution. The resizing behavior depends on two parameters: doresize and size. Videos should not be upsampled. If doresize is set to True, the model resizes images so that the longest edge is 4512 pixels by default.

**Target**

입력된 이미지는 설정에 따라 원본 해상도를 유지하거나 크기를 조절할 수 있습니다. 이때 이미지 크기 조절 여부와 방식은 doresize와 size 파라미터로 결정됩니다. 만약 doresize가 True일 경우, 모델은 기본적으로 이미지의 가장 긴 변을 4512 픽셀이 되도록 크기를 조절합니다. 이 기본 동작은 size 파라미터에 딕셔너리를 전달하여 원하는 값으로 직접 설정할 수 있습니다. 예를 들어, 기본값은 {"longestedge": 4 512} 이여도 사용자 필요에 따라 다른 값으로 변경할 수 있습니다.

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

## pipeline_webserver.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/pipeline_webserver.md

### chunk 1
- source_chars: 913
- target_chars: 615
- length_ratio: 1.48

**Source**

A web server is a system that waits for requests and serves them as they come in. This means you can use [Pipeline] as an inference engine on a web server, since you can use an iterator (similar to how you would iterate over a dataset) to handle each incoming request. Designing a web server with [Pipeline] is unique though because they're fundamentally different. Web servers are multiplexed (multithreaded, async, etc.) to handle multiple requests concurrently. [Pipeline] and its underlying model on the other hand are not designed for parallelism because they take a lot of memory. It's best to give a [Pipeline] all the available resources when they're running or for a compute intensive job. This guide shows how to work around this difference by using a web server to handle the lighter load of receiving and sending requests, and having a single thread to handle the heavier load of runni...

**Target**

추론 엔진을 만드는 것은 복잡한 주제이며, "최선의" 솔루션은 문제 공간에 따라 달라질 가능성이 높습니다. CPU 또는 GPU를 사용하는지에 따라 다르고 낮은 지연 시간을 원하는지, 높은 처리량을 원하는지, 다양한 모델을 지원할 수 있길 원하는지, 하나의 특정 모델을 고도로 최적화하길 원하는지 등에 따라 달라집니다. 이 주제를 해결하는 방법에는 여러 가지가 있으므로, 이 장에서 제시하는 것은 처음 시도해 보기에 좋은 출발점일 수는 있지만, 이 장을 읽는 여러분이 필요로 하는 최적의 솔루션은 아닐 수 있습니다. 핵심적으로 이해해야 할 점은 dataset를 다룰 때와 마찬가지로 반복자를 사용 가능하다는 것입니다. 왜냐하면, 웹 서버는 기본적으로 요청을 기다리고 들어오는 대로 처리하는 시스템이기 때문입니다. 보통 웹 서버는 다양한 요청을 동시에 다루기 위해 매우 다중화된 구조(멀티 스레딩, 비동기 등)를 지니고 있습니다. 반면에, 파이프라인(대부분 파이프라인 안에 있는 모델)은 병렬처리에 그다지 좋지 않습니다. 왜냐하면 파이프라인은 많은 RAM을 차지하기 때문입니다. 따라서, 파이프라인이 실행 중이거나 계산 집약적인 작업 중일 때 모든 사용 가능한 리소스를 제공하는 것이 가장 좋습니다.

### chunk 2
- source_chars: 229
- target_chars: 228
- length_ratio: 1.0

**Source**

Starlette is a lightweight framework for building web servers. You can use any other framework you'd like, but you may have to make some changes to the code below. Before you begin, make sure Starlette and uvicorn are installed.

**Target**

이 문제를 우리는 웹 서버가 요청을 받고 보내는 가벼운 부하를 처리하고, 실제 작업을 처리하는 단일 스레드를 갖는 방법으로 해결할 것입니다. 이 예제는 starlette 라이브러리를 사용합니다. 실제 프레임워크는 중요하지 않지만, 다른 프레임워크를 사용하는 경우 동일한 효과를 보기 위해선 코드를 조정하거나 변경해야 할 수 있습니다. 자, 이제 웹 서버를 만드는 방법에 대한 좋은 개념을 알게 되었습니다!

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

## tasks/asr.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/asr.md

### chunk 1
- source_chars: 417
- target_chars: 254
- length_ratio: 1.64

**Source**

Automatic speech recognition (ASR) converts a speech signal to text, mapping a sequence of audio inputs to text outputs. Virtual assistants like Siri and Alexa use ASR models to help users every day, and there are many other useful user-facing applications like live captioning and note-taking during meetings. This guide will show you how to: Fine-tune Wav2Vec2 on the MInDS-14 dataset to transcribe audio to text.

**Target**

자동 음성 인식(Automatic Speech Recognition, ASR)은 음성 신호를 텍스트로 변환하여 음성 입력 시퀀스를 텍스트 출력에 매핑합니다. Siri와 Alexa와 같은 가상 어시스턴트는 ASR 모델을 사용하여 일상적으로 사용자를 돕고 있으며, 회의 중 라이브 캡션 및 메모 작성과 같은 유용한 사용자 친화적 응용 프로그램도 많이 있습니다. MInDS-14 데이터 세트에서 Wav2Vec2를 미세 조정하여 오디오를 텍스트로 변환합니다.

### chunk 2
- source_chars: 222
- target_chars: 160
- length_ratio: 1.39

**Source**

Use your fine-tuned model for inference. To see all architectures and checkpoints compatible with this task, we recommend checking the task-page Before you begin, make sure you have all the necessary libraries installed:

**Target**

이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 작업 페이지를 확인하는 것이 좋습니다. 시작하기 전에 필요한 모든 라이브러리가 설치되어 있는지 확인하세요: Hugging Face 계정에 로그인하면 모델을 업로드하고 커뮤니티에 공유할 수 있습니다. 토큰을 입력하여 로그인하세요.

## tasks/keypoint_detection.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/keypoint_detection.md

### chunk 1
- source_chars: 494
- target_chars: 201
- length_ratio: 2.46

**Source**

Keypoint detection identifies and locates specific points of interest within an image. These keypoints, also known as landmarks, represent meaningful features of objects, such as facial features or object parts. These models take an image input and return the following outputs: Keypoints and Scores: Points of interest and their confidence scores. Descriptors: A representation of the image region surrounding each keypoint, capturing its texture, gradient, orientation and other properties.

**Target**

키포인트 감지(Keypoint detection)은 이미지 내의 특정 포인트를 식별하고 위치를 탐지합니다. 이러한 키포인트는 랜드마크라고도 불리며 얼굴 특징이나 물체의 일부와 같은 의미 있는 특징을 나타냅니다. 키포인트 감지 모델들은 이미지를 입력으로 받아 아래와 같은 출력을 반환합니다. 키포인트들과 점수: 관심 포인트들과 해당 포인트에 대한 신뢰도 점수

### chunk 2
- source_chars: 195
- target_chars: 172
- length_ratio: 1.13

**Source**

In this guide, we will show how to extract keypoints from images. For this tutorial, we will use SuperPoint, a foundation model for keypoint detection. Let's test the model on the images below.

**Target**

디스크립터(Descriptors): 각 키포인트를 둘러싼 이미지 영역의 표현으로 텍스처, 그라데이션, 방향 및 기타 속성을 캡처합니다. 이번 가이드에서는 이미지에서 키포인트를 추출하는 방법을 다루어 보겠습니다. 이번 튜토리얼에서는 키포인트 감지의 기본이 되는 모델인 SuperPoint를 사용해보겠습니다.

## tasks/prompting.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/prompting.md

### chunk 1
- source_chars: 464
- target_chars: 438
- length_ratio: 1.06

**Source**

Prompt engineering or prompting, uses natural language to improve large language model (LLM) performance on a variety of tasks. A prompt can steer the model towards generating a desired output. In many cases, you don't even need a fine-tuned model for a task. You just need a good prompt. Try prompting a LLM to classify some text. When you create a prompt, it's important to provide very specific instructions about the task and what the result should look like.

**Target**

Falcon, LLaMA 등의 대규모 언어 모델은 사전 훈련된 트랜스포머 모델로, 초기에는 주어진 입력 텍스트에 대해 다음 토큰을 예측하도록 훈련됩니다. 이들은 보통 수십억 개의 매개변수를 가지고 있으며, 장기간에 걸쳐 수조 개의 토큰으로 훈련됩니다. 그 결과, 이 모델들은 매우 강력하고 다재다능해져서, 자연어 프롬프트로 모델에 지시하여 다양한 자연어 처리 작업을 즉시 수행할 수 있습니다. 최적의 출력을 보장하기 위해 이러한 프롬프트를 설계하는 것을 흔히 "프롬프트 엔지니어링"이라고 합니다. 프롬프트 엔지니어링은 상당한 실험이 필요한 반복적인 과정입니다. 자연어는 프로그래밍 언어보다 훨씬 유연하고 표현력이 풍부하지만, 동시에 모호성을 초래할 수 있습니다. 또한, 자연어 프롬프트는 변화에 매우 민감합니다. 프롬프트의 사소한 수정만으로도 완전히 다른 출력이 나올 수 있습니다.

### chunk 2
- source_chars: 266
- target_chars: 244
- length_ratio: 1.09

**Source**

The challenge lies in designing prompts that produces the results you're expecting because language is so incredibly nuanced and expressive. This guide covers prompt engineering best practices, techniques, and examples for how to solve language and reasoning tasks.

**Target**

모든 경우에 적용할 수 있는 정확한 프롬프트 생성 공식은 없지만, 연구자들은 더 일관되게 최적의 결과를 얻는 데 도움이 되는 여러 가지 모범 사례를 개발했습니다. 이 가이드에서는 더 나은 대규모 언어 모델 프롬프트를 작성하고 다양한 자연어 처리 작업을 해결하는 데 도움이 되는 프롬프트 엔지니어링 모범 사례를 다룹니다: 고급 프롬프팅 기법: 퓨샷(Few-shot) 프롬프팅과 생각의 사슬(Chain-of-thought, CoT) 기법

## tasks/sequence_classification.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/sequence_classification.md

### chunk 1
- source_chars: 490
- target_chars: 285
- length_ratio: 1.72

**Source**

Text classification is a common NLP task that assigns a label or class to text. Some of the largest companies run text classification in production for a wide range of practical applications. One of the most popular forms of text classification is sentiment analysis, which assigns a label like 🙂 positive, 🙁 negative, or 😐 neutral to a sequence of text. This guide will show you how to: Finetune DistilBERT on the IMDb dataset to determine whether a movie review is positive or negative.

**Target**

텍스트 분류는 자연어 처리의 일종으로, 텍스트에 레이블 또는 클래스를 지정하는 작업입니다. 많은 대기업이 다양한 실용적인 응용 분야에서 텍스트 분류를 운영하고 있습니다. 가장 인기 있는 텍스트 분류 형태 중 하나는 감성 분석으로, 텍스트 시퀀스에 🙂 긍정, 🙁 부정 또는 😐 중립과 같은 레이블을 지정합니다. IMDb 데이터셋에서 DistilBERT를 파인 튜닝하여 영화 리뷰가 긍정적인지 부정적인지 판단합니다. 이 작업과 호환되는 모든 아키텍처와 체크포인트를 보려면 작업 페이지를 확인하는 것이 좋습니다.

### chunk 3
- source_chars: 338
- target_chars: 161
- length_ratio: 2.1

**Source**

To see all architectures and checkpoints compatible with this task, we recommend checking the task-page. Before you begin, make sure you have all the necessary libraries installed: We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login:

**Target**

Hugging Face 계정에 로그인하여 모델을 업로드하고 커뮤니티에 공유하는 것을 권장합니다. 메시지가 표시되면, 토큰을 입력하여 로그인하세요: 먼저 🤗 Datasets 라이브러리에서 IMDb 데이터셋을 가져옵니다: label: 0은 부정적인 리뷰, 1은 긍정적인 리뷰를 나타냅니다.

## training.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/training.md

### chunk 1
- source_chars: 421
- target_chars: 322
- length_ratio: 1.31

**Source**

Fine-tuning continues training a large pretrained model on a smaller dataset specific to a task or domain. For example, fine-tuning on a dataset of coding examples helps the model get better at coding. Fine-tuning is identical to pretraining except you don't start with random weights. It also requires far less compute, data, and time. The tutorial below walks through fine-tuning a large language model with [Trainer].

**Target**

사전 학습된 모델을 사용하면 상당한 이점이 있습니다. 계산 비용과 탄소발자국을 줄이고, 처음부터 모델을 학습시킬 필요 없이 최신 모델을 사용할 수 있습니다. 🤗 Transformers는 다양한 작업을 위해 사전 학습된 수천 개의 모델에 액세스할 수 있습니다. 사전 학습된 모델을 사용하는 경우, 자신의 작업과 관련된 데이터셋을 사용해 학습합니다. 이것은 미세 튜닝이라고 하는 매우 강력한 훈련 기법입니다. 이 튜토리얼에서는 당신이 선택한 딥러닝 프레임워크로 사전 학습된 모델을 미세 튜닝합니다: 🤗 Transformers로 사전 학습된 모델 미세 튜닝하기 [Trainer].

### chunk 2
- source_chars: 197
- target_chars: 191
- length_ratio: 1.03

**Source**

Log in to your Hugging Face account with your user token to push your fine-tuned model to the Hub. Load a dataset and tokenize the text column the model trains on (horoscope in the dataset below).

**Target**

Keras를 사용하여 TensorFlow에서 사전 학습된 모델을 미세 튜닝하기. 기본 PyTorch에서 사전 학습된 모델을 미세 튜닝하기. 사전 학습된 모델을 미세 튜닝하기 위해서 데이터셋을 다운로드하고 훈련할 수 있도록 준비하세요. 이전 튜토리얼에서 훈련을 위해 데이터를 처리하는 방법을 보여드렸는데, 지금이 배울 걸 되짚을 기회입니다!

## troubleshooting.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/troubleshooting.md

### chunk 1
- source_chars: 767
- target_chars: 438
- length_ratio: 1.75

**Source**

Sometimes errors occur, but we are here to help! This guide covers some of the most common issues we've seen and how you can resolve them. However, this guide isn't meant to be a comprehensive collection of every 🤗 Transformers issue. For more help with troubleshooting your issue, try: Asking for help on the forums. There are specific categories you can post your question to, like Beginners or 🤗 Transformers. Make sure you write a good descriptive forum post with some reproducible code to maximize the likelihood that your problem is solved! Create an Issue on the 🤗 Transformers repository if it is a bug related to the library. Try to include as much information describing the bug as possible to help us better figure out what's wrong and how we can fix it.

**Target**

때때로 오류가 발생할 수 있지만, 저희가 도와드리겠습니다! 이 가이드는 현재까지 확인된 가장 일반적인 문제 몇 가지와 그것들을 해결하는 방법에 대해 다룹니다. 그러나 이 가이드는 모든 🤗 Transformers 문제를 포괄적으로 다루고 있지 않습니다. 문제 해결에 더 많은 도움을 받으려면 다음을 시도해보세요: 포럼에서 도움을 요청하세요. Beginners 또는 🤗 Transformers와 같은 특정 카테고리에 질문을 게시할 수 있습니다. 재현 가능한 코드와 함께 잘 서술된 포럼 게시물을 작성하여 여러분의 문제가 해결될 가능성을 극대화하세요! 라이브러리와 관련된 버그이면 🤗 Transformers 저장소에서 이슈를 생성하세요. 버그에 대해 설명하는 정보를 가능한 많이 포함하려고 노력하여, 무엇이 잘못 되었는지와 어떻게 수정할 수 있는지 더 잘 파악할 수 있도록 도와주세요.

### chunk 2
- source_chars: 505
- target_chars: 276
- length_ratio: 1.83

**Source**

Check the Migration guide if you use an older version of 🤗 Transformers since some important changes have been introduced between versions. For more details about troubleshooting and getting help, take a look at Chapter 8 of the Hugging Face course. Some GPU instances on cloud and intranet setups are firewalled to external connections, resulting in a connection error. When your script attempts to download model weights or datasets, the download will hang and then timeout with the following message:

**Target**

이전 버전의 🤗 Transformers을 사용하는 경우 중요한 변경 사항이 버전 사이에 도입되었기 때문에 마이그레이션 가이드를 확인하세요. 문제 해결 및 도움 매뉴얼에 대한 자세한 내용은 Hugging Face 강좌의 8장을 참조하세요. 클라우드 및 내부망(intranet) 설정의 일부 GPU 인스턴스는 외부 연결에 대한 방화벽으로 차단되어 연결 오류가 발생할 수 있습니다. 스크립트가 모델 가중치나 데이터를 다운로드하려고 할 때, 다운로드가 중단되고 다음 메시지와 함께 시간 초과됩니다:
