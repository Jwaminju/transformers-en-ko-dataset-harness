# Clean Suspect Review

- documents: 10

## troubleshooting.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/troubleshooting.md
- max_ratio: 2.97
- avg_ratio: 1.5

### chunk 6
- ratio: 2.97
- source_chars: 635
- target_chars: 214

**Source**

You should try to run the code on a CPU first to get a more descriptive error message. Add the following environment variable to the beginning of your code to switch to a CPU: Another option is to get a better traceback from the GPU. Add the following environment variable to the beginning of your code to get the traceback to point to the source of the error: In some cases, the output hiddenstate may be incorrect if the inputids include padding tokens. To demonstrate, load a model and tokenizer. You can access a model's padtokenid to see its value. The padtokenid may be None for some models, but you can always manually set it.

**Target**

모델을 [~TFPretrainedModel.savepretrained]로 저장하고 [~TFPreTrainedModel.frompretrained]로 다시 가져옵니다: 특히 최신 모델인 경우 만날 수 있는 다른 일반적인 오류는 ImportError입니다: 이러한 오류 유형의 경우 최신 모델에 액세스할 수 있도록 최신 버전의 🤗 Transformers가 설치되어 있는지 확인하세요:

### chunk 2
- ratio: 1.83
- source_chars: 505
- target_chars: 276

**Source**

Check the Migration guide if you use an older version of 🤗 Transformers since some important changes have been introduced between versions. For more details about troubleshooting and getting help, take a look at Chapter 8 of the Hugging Face course. Some GPU instances on cloud and intranet setups are firewalled to external connections, resulting in a connection error. When your script attempts to download model weights or datasets, the download will hang and then timeout with the following message:

**Target**

이전 버전의 🤗 Transformers을 사용하는 경우 중요한 변경 사항이 버전 사이에 도입되었기 때문에 마이그레이션 가이드를 확인하세요. 문제 해결 및 도움 매뉴얼에 대한 자세한 내용은 Hugging Face 강좌의 8장을 참조하세요. 클라우드 및 내부망(intranet) 설정의 일부 GPU 인스턴스는 외부 연결에 대한 방화벽으로 차단되어 연결 오류가 발생할 수 있습니다. 스크립트가 모델 가중치나 데이터를 다운로드하려고 할 때, 다운로드가 중단되고 다음 메시지와 함께 시간 초과됩니다:

## model_doc/smolvlm.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/smolvlm.md
- max_ratio: 2.77
- avg_ratio: 1.63

### chunk 3
- ratio: 2.77
- source_chars: 560
- target_chars: 202

**Source**

The default resizing behavior can be customized by passing a dictionary to the size parameter. For example, {"longestedge": 4 512} is the default, but you can change it to a different value if needed. Here's how to control resizing and set a custom size: Additionally, the maximagesize parameter, which controls the size of each square patch the image is decomposed into, is set to 512 by default but can be adjusted as needed. After resizing (if applicable), the image processor decomposes the images into square patches based on the maximagesize parameter.

**Target**

다음은 리사이징을 제어하고 사용자 정의 크기로 변경하는 방법입니다: 또한, maximagesize 매개변수는 이미지를 분할하는 정사각형 패치의 크기를 제어합니다. 이 값은 기본적으로 512로 설정되어 있으며 필요에 따라 조정 가능합니다. 이미지 처리기는 리사이징을 마친 후, maximagesize 값을 기준으로 이미지를 여러 개의 정사각형 패치로 분할합니다.

### chunk 4
- ratio: 1.68
- source_chars: 276
- target_chars: 164

**Source**

This model was contributed by orrzohar. The model can accept both images and videos as input, but you should use only one of the modalities at a time. Here's an example code for that. The model can batch inputs composed of several images/videos and text. Here is an example.

**Target**

이 모델은 이미지와 비디오를 모두 입력으로 받을 수 있지만, 한 번에 사용할 수 있는 미디어는 반드시 하나의 종류여야 합니다. 관련 예시 코드는 다음과 같습니다. 이 모델은 여러 이미지, 비디오, 텍스트로 구성된 입력을 한 번에 배치 형태로 처리할 수 있습니다. 관련 예시는 다음과 같습니다.

## pipeline_webserver.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/pipeline_webserver.md
- max_ratio: 2.76
- avg_ratio: 1.37

### chunk 6
- ratio: 2.76
- source_chars: 530
- target_chars: 192

**Source**

The example below is written in pseudocode for readability rather than performance, in particular, you'll notice that: The timeout is reset on every queue fetch, so you could end up waiting much longer than the timeout value before processing a request. This would also delay the first inference request by that amount of time. The web server always waits 1ms even if the queue is empty, which is inefficient, because that time can be used to start inference. It could make sense though if batching is essential to your use case.

**Target**

큐에 아무것도 없을 때 추론을 원하는 경우에는 최선의 방법이 아닐 수 있습니다. 하지만 배치 작업이 사용례에 따라 정말로 중요하다면 의미가 있을 수도 있습니다. 메모리가 모자라거나, 공간이 부족하거나, 모델을 가져오는 데에 실패하거나, 쿼리가 잘못되었거나, 쿼리는 정확해도 모델 설정이 잘못되어 실행에 실패하는 등등 많은 경우가 존재합니다.

### chunk 11
- ratio: 1.93
- source_chars: 552
- target_chars: 286

**Source**

It is relatively simple to implement these error types since it's only a single queue. Take a look at the queue size to determine when to start returning errors before your server fails under load. PyTorch is not async aware, so computation will block the main thread from running. For this reason, it's better to run PyTorch on its own separate thread or process. When inference of a single request is especially long (more than 1s), it's even more important because it means every query during inference must wait 1s before even receiving an error.

**Target**

여기서는 이 작업이 수행되지 않았습니다. 왜냐하면 코드가 훨씬 더 복잡하기 때문입니다(주로 스레드, 비동기 처리, 큐가 서로 잘 맞지 않기 때문입니다). 단일 항목의 추론이 오래 걸린다면 (> 1초), 메인 쓰레드를 차단하는 것은 중요할 수 있습니다. 왜냐하면 이 경우 추론 중 모든 쿼리는 오류를 받기 전에 1초를 기다려야 하기 때문입니다. 일반적으로, 배치 처리가 1개 항목을 한 번에 전달하는 것에 비해 반드시 성능 향상이 있는 것은 아닙니다(자세한 내용은 batching details을 참고하세요).

## tasks/sequence_classification.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/sequence_classification.md
- max_ratio: 2.57
- avg_ratio: 1.92

### chunk 9
- ratio: 2.57
- source_chars: 544
- target_chars: 212

**Source**

Define your training hyperparameters in [TrainingArguments]. The only required parameter is outputdir which specifies where to save your model. You'll push this model to the Hub by setting pushtohub=True (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [Trainer] will evaluate the accuracy and save the training checkpoint. Pass the training arguments to [Trainer] along with the model, dataset, tokenizer, data collator, and computemetrics function. Call [~Trainer.train] to finetune your model.

**Target**

[TrainingArguments]에서 하이퍼파라미터를 정의하세요. outputdir는 모델을 저장할 위치를 지정하는 유일한 파라미터입니다. 이 모델을 Hub에 업로드하기 위해 pushtohub=True를 설정합니다. (모델을 업로드하기 위해 Hugging Face에 로그인해야합니다.) 각 에폭이 끝날 때마다, [Trainer]는 정확도를 평가하고 훈련 체크포인트를 저장합니다.

### chunk 12
- ratio: 2.56
- source_chars: 305
- target_chars: 119

**Source**

Grab some text you'd like to run inference on: The simplest way to try out your finetuned model for inference is to use it in a [pipeline]. Instantiate a pipeline for sentiment analysis with your model, and pass your text to it: You can also manually replicate the results of the pipeline if you'd like:

**Target**

텍스트 분류를 위한 모델을 파인 튜닝하는 자세한 예제는 다음 PyTorch notebook 또는 TensorFlow notebook를 참조하세요. 좋아요, 이제 모델을 파인 튜닝했으니 추론에 사용할 수 있습니다!

## tasks/asr.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/asr.md
- max_ratio: 2.52
- avg_ratio: 1.78

### chunk 6
- ratio: 2.52
- source_chars: 486
- target_chars: 193

**Source**

The MInDS-14 dataset has a sampling rate of 8000Hz (you can find this information in its dataset card), which means you'll need to resample the dataset to 16000Hz to use the pretrained Wav2Vec2 model: As you can see in the transcription above, the text contains a mix of uppercase and lowercase characters. The Wav2Vec2 tokenizer is only trained on uppercase characters so you'll need to make sure the text matches the tokenizer's vocabulary: Now create a preprocessing function that:

**Target**

위의 'transcription'에서 볼 수 있듯이 텍스트는 대문자와 소문자가 섞여 있습니다. Wav2Vec2 토크나이저는 대문자 문자에 대해서만 훈련되어 있으므로 텍스트가 토크나이저의 어휘와 일치하는지 확인해야 합니다: 이제 다음 작업을 수행할 전처리 함수를 만들어보겠습니다: audio 열을 호출하여 오디오 파일을 가져오고 리샘플링합니다.

### chunk 3
- ratio: 2.49
- source_chars: 474
- target_chars: 190

**Source**

We encourage you to login to your Hugging Face account so you can upload and share your model with the community. When prompted, enter your token to login: Start by loading a smaller subset of the MInDS-14 dataset from the 🤗 Datasets library. This will give you a chance to experiment and make sure everything works before spending more time training on the full dataset. Split the dataset's train split into a train and test set with the [~Dataset.traintestsplit] method:

**Target**

먼저, 🤗 Datasets 라이브러리에서 MInDS-14 데이터 세트의 일부분을 가져오세요. 이렇게 하면 전체 데이터 세트에 대한 훈련에 시간을 들이기 전에 모든 것이 작동하는지 실험하고 검증할 수 있습니다. [~Dataset.traintestsplit] 메소드를 사용하여 데이터 세트의 train을 훈련 세트와 테스트 세트로 나누세요:

## fsdp.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/fsdp.md
- max_ratio: 2.52
- avg_ratio: 1.52

### chunk 6
- ratio: 2.52
- source_chars: 735
- target_chars: 292

**Source**

FSDP is applied by wrapping each layer in the network. The wrapping is usually applied in a nested way where the full weights are discarded after each forward pass to save memory for the next layer. There are several wrapping policies available, but the auto wrapping policy is the simplest and doesn't require any changes to your code. Specify fsdpautowrappolicy: TRANSFORMERBASEDWRAP to wrap a Transformer layer and fsdptransformerlayerclstowrap to determine which layer to wrap (for example, BertLayer). Size-based wrapping is also available. If a layer exceeds a certain number of parameters, it is wrapped. Specify fsdpwrappolicy: SIZEDBASEDWRAP and minnumparam to set the minimum number of parameters for a layer to be wrapped.

**Target**

FSDP는 네트워크의 각 레이어를 래핑하여 적용됩니다. 래핑은 일반적으로 중첩 방식으로 적용되며 각각 순방향으로 지나간 후 전체 가중치를 삭제하여 다음 레이어에서 사용할 메모리를 절약합니다. 자동 래핑 정책은 이를 구현하는 가장 간단한 방법이며 코드를 변경할 필요가 없습니다. Transformer 레이어를 래핑하려면 fsdpautowrappolicy: TRANSFORMERBASEDWRAP를 선택하고 래핑할 레이어를 지정하려면 fsdptransformerlayerclstowrap를 선택하세요 (예: BertLayer).

### chunk 8
- ratio: 2.42
- source_chars: 651
- target_chars: 269

**Source**

PyTorch XLA, a package for running PyTorch on XLA devices, enables FSDP on TPUs. Modify the configuration file to include the parameters below. Refer to the xlafsdpsettings parameter for additional XLA-specific parameters you can configure for FSDP. After running accelerate config, your configuration file should be ready. An example configuration file is shown below that fully shards the parameter, gradient and optimizer states on two GPUs. Your file may look different depending on how you set up your configuration. Run the accelerate launch command to launch a training script with the FSDP configurations you chose in the configuration file.

**Target**

그러나 훈련이 끝나면 전체 상태 딕셔너리를 저장해야 합니다. 분할된 상태 딕셔너리는 FSDP와만 호환되기 때문입니다. PyTorch XLA는 TPU에 대한 FSDP 훈련을 지원하며 accelerate config로 생성된 FSDP 구성 파일을 수정하여 활성화할 수 있습니다. 위에서 지정한 분할 전략 및 래핑 옵션 외에도 아래에 표시된 매개변수를 파일에 추가할 수 있습니다. xlafsdpsettings는 FSDP에 대한 추가적인 XLA 특정 매개변수를 구성할 수 있게 합니다.

## tasks/keypoint_detection.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/tasks/keypoint_detection.md
- max_ratio: 2.46
- avg_ratio: 1.84

### chunk 1
- ratio: 2.46
- source_chars: 494
- target_chars: 201

**Source**

Keypoint detection identifies and locates specific points of interest within an image. These keypoints, also known as landmarks, represent meaningful features of objects, such as facial features or object parts. These models take an image input and return the following outputs: Keypoints and Scores: Points of interest and their confidence scores. Descriptors: A representation of the image region surrounding each keypoint, capturing its texture, gradient, orientation and other properties.

**Target**

키포인트 감지(Keypoint detection)은 이미지 내의 특정 포인트를 식별하고 위치를 탐지합니다. 이러한 키포인트는 랜드마크라고도 불리며 얼굴 특징이나 물체의 일부와 같은 의미 있는 특징을 나타냅니다. 키포인트 감지 모델들은 이미지를 입력으로 받아 아래와 같은 출력을 반환합니다. 키포인트들과 점수: 관심 포인트들과 해당 포인트에 대한 신뢰도 점수

### chunk 3
- ratio: 1.94
- source_chars: 384
- target_chars: 198

**Source**

We can now process our inputs and infer. The model output has relative keypoints, descriptors, masks and scores for each item in the batch. The mask highlights areas of the image where keypoints are present. To plot actual keypoints in the image, we need to postprocess the output. To do so, we have to pass the actual image sizes to postprocesskeypointdetection along with outputs.

**Target**

모델 출력에는 배치 내의 각 항목에 대한 상대적인 키포인트, 디스크립터, 마스크와 점수가 있습니다. 마스크는 이미지에서 키포인트가 있는 영역을 강조하는 역할을 합니다. 이미지에 실제 키포인트를 표시하기 위해선 결과값을 후처리 해야합니다. 이를 위해 실제 이미지 크기를 결과값과 함께 postprocesskeypointdetection에 전달해야 합니다.

## model_doc/albert.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/albert.md
- max_ratio: 2.41
- avg_ratio: 1.51

### chunk 4
- ratio: 2.41
- source_chars: 878
- target_chars: 365

**Source**

The embedding size E is different from the hidden size H because the embeddings are context independent (one embedding vector represents one token) and the hidden states are context dependent (one hidden state represents a sequence of tokens). The embedding matrix is also larger because V x E where V is the vocabulary size. As a result, it's more logical if H >> E. If E < H, the model has less parameters. The resources provided in the following sections consist of a list of official Hugging Face and community (indicated by 🌎) resources to help you get started with AlBERT. If you're interested in submitting a resource to be included here, please feel free to open a Pull Request and we'll review it! The resource should ideally demonstrate something new instead of duplicating an existing resource. [AlbertForSequenceClassification] is supported by this example script.

**Target**

BERT는 절대 위치 임베딩을 사용하므로, 오른쪽에 입력이 패딩돼야 합니다. 임베딩 크기 E는 히든 크기 H와 다릅니다. 임베딩은 문맥에 독립적(각 토큰마다 하나의 임베딩 벡터)이고, 은닉 상태는 문맥에 의존적(토큰 시퀀스마다 하나의 은닉 상태)입니다. 임베딩 행렬은 V x E(V: 어휘 크기)이므로, 일반적으로 H >> E가 더 논리적입니다. E < H일 때 모델 파라미터가 더 적어집니다. 아래 섹션의 자료들은 공식 Hugging Face 및 커뮤니티(🌎 표시) 자료로, AlBERT를 시작하는 데 도움이 됩니다. 여기에 추가할 자료가 있다면 Pull Request를 보내주세요! 기존 자료와 중복되지 않고 새로운 내용을 담고 있으면 좋습니다.

### chunk 1
- ratio: 1.94
- source_chars: 851
- target_chars: 439

**Source**

ALBERT is designed to address memory limitations of scaling and training of BERT. It adds two parameter reduction techniques. The first, factorized embedding parametrization, splits the larger vocabulary embedding matrix into two smaller matrices so you can grow the hidden size without adding a lot more parameters. The second, cross-layer parameter sharing, allows layer to share parameters which keeps the number of learnable parameters lower. ALBERT was created to address problems like -- GPU/TPU memory limitations, longer training times, and unexpected model degradation in BERT. ALBERT uses two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT: Factorized embedding parameterization: The large vocabulary embedding matrix is decomposed into two smaller matrices, reducing memory consumption.

**Target**

ALBERT는 BERT의 확장성과 학습 시 메모리 한계를 해결하기 위해 설계된 모델입니다. 이 모델은 두 가지 파라미터 감소 기법을 도입합니다. 첫 번째는 임베딩 행렬 분해(factorized embedding parametrization)로, 큰 어휘 임베딩 행렬을 두 개의 작은 행렬로 분해하여 히든 사이즈를 늘려도 파라미터 수가 크게 증가하지 않도록 합니다. 두 번째는 계층 간 파라미터 공유(cross-layer parameter sharing)로, 여러 계층이 파라미터를 공유하여 학습해야 할 파라미터 수를 줄입니다. ALBERT는 BERT에서 발생하는 GPU/TPU 메모리 한계, 긴 학습 시간, 갑작스런 성능 저하 문제를 해결하기 위해 만들어졌습니다. ALBERT는 파라미터를 줄이기 위해 두 가지 기법을 사용하여 메모리 사용량을 줄이고 BERT의 학습 속도를 높입니다:

## model_doc/llama3.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/llama3.md
- max_ratio: 2.33
- avg_ratio: 1.62

### chunk 5
- ratio: 2.33
- source_chars: 1014
- target_chars: 436

**Source**

The architecture is exactly the same as Llama2. The tokenizer is a BPE model based on tiktoken (vs the one based on sentencepiece implementation for Llama2). The main difference that it ignores BPE merge rules when an input token is part of the vocab. This means that if no merge exist to produce "hugging", instead of having the smallest units, like ["hug","ging"] form 2 tokens, if "hugging" is part of the vocab, it will be automatically returned as a token. The original model uses padid = -1 which means that there is no padding token. We can't have the same logic, make sure to add a padding token using tokenizer.addspecialtokens({"padtoken":" "}) and resize the token embedding accordingly. You should also set the model.config.padtokenid. The embedtokens layer of the model is initialized with self.embedtokens = nn.Embedding(config.vocabsize, config.hiddensize, self.config.paddingidx), ...

**Target**

기본 모델은 패딩 토큰이 없다는 것을 의미하는 padid = -1을 사용합니다. 같은 로직을 사용할 수 없으니 tokenizer.addspecialtokens({"padtoken":" "})를 사용하여 토큰을 추가하고 임베딩 크기도 확실히 조정해야 합니다. model.config.padtokenid도 설정이 필요합니다. 모델의 embedtokens 레이어는 self.embedtokens = nn.Embedding(config.vocabsize, config.hiddensize, self.config.paddingidx)로 초기화되며, 패딩 토큰을 인코딩하는 것이 0(zero)를 출력하게 할 것인지 그래서 초기화가 추천될때 이를 통화시킬 것인지를 정하게 합니다. 원본 체크포인트는 이 변환 스크립트를 이용해서 변환 가능합니다. 스크립트는 다음 명령어로 호출할 수 있습니다:

### chunk 7
- ratio: 1.93
- source_chars: 697
- target_chars: 362

**Source**

come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM). For the 75B model, it's thus 145GB of RAM needed. When using Flash Attention 2 via attnimplementation="flashattention2", don't pass dtype to the frompretrained class method and use Automatic Mixed-Precision training. When using Trainer, it is simply specifying either fp16 or bf16 to True. Otherwise, make sure you are using torch.autocast. This is required because the Flash Attention only support fp16 and bf16 data type. A ton of cool resources are already available on the documentation page of Llama2, inviting contributors to add new resources curated for Llama3 here! 🤗

**Target**

attnimplementation="flashattention2"를 통해서 플래시 어텐션2를 사용할 때, frompretrained 클래스 메서드에 dtype를 전달하지 말고 자동 혼합 정밀도(Automatic Mixed-Precision) 학습을 사용하세요. Trainer를 사용할 때는 단순히 fp16 또는 bf16을 True로 설정하면 됩니다. 그렇지 않으면 반드시 torch.autocast를 사용해야 합니다. 플래시 어텐션은 fp16과 bf16 데이터 유형만 지원하기 때문입니다. 라마2 문서 페이지에서는 이미 수 많은 멋지고 유익한 자료들을 제공하고 있습니다. 이곳에 라마3에 대한 새로운 자료를 더해주실 컨트리뷰터들을 초대합니다! 🤗

## model_doc/autoformer.md
- link: https://github.com/huggingface/transformers/blob/main/docs/source/ko/model_doc/autoformer.md
- max_ratio: 2.31
- avg_ratio: 1.97

### chunk 2
- ratio: 2.31
- source_chars: 1567
- target_chars: 678

**Source**

The abstract from the paper is the following: Extending the forecasting time is a critical demand for real applications, such as extreme weather early warning and long-term energy consumption planning. This paper studies the long-term forecasting problem of time series. Prior Transformer-based models adopt various self-attention mechanisms to discover the long-range dependencies. However, intricate temporal patterns of the long-term future prohibit the model from finding reliable dependencies. Also, Transformers have to adopt the sparse versions of point-wise self-attentions for long series efficiency, resulting in the information utilization bottleneck. Going beyond Transformers, we design Autoformer as a novel decomposition architecture with an Auto-Correlation mechanism. We break with the pre-processing convention of series decomposition and renovate it as a basic inner block of de...

**Target**

예측 시간을 연장하는 것은 극한 기상 조기 경보 및 장기 에너지 소비 계획과 같은 실제 응용 프로그램에 중요한 요구 사항입니다. 본 논문은 시계열의 장기 예측 문제를 연구합니다. 기존의 트랜스포머 기반 모델들은 장거리 종속성을 발견하기 위해 다양한 셀프 어텐션 메커니즘을 채택합니다. 그러나 장기 미래의 복잡한 시간적 패턴으로 인해 모델이 신뢰할 수 있는 종속성을 찾기 어렵습니다. 또한, 트랜스포머는 긴 시계열의 효율성을 위해 점별 셀프 어텐션의 희소 버전을 채택해야 하므로 정보 활용의 병목 현상이 발생합니다. 우리는 트랜스포머를 넘어서 자기상관 메커니즘을 갖춘 새로운 분해 아키텍처인 Autoformer를 설계했습니다. 우리는 시계열 분해의 전처리 관행을 깨고 이를 심층 모델의 기본 내부 블록으로 혁신했습니다. 이 설계는 Autoformer에 복잡한 시계열에 대한 점진적 분해 능력을 부여합니다. 또한, 확률 과정 이론에서 영감을 받아 시계열의 주기성을 기반으로 자기상관 메커니즘을 설계했으며, 이는 하위 시계열 수준에서 종속성 발견과 표현 집계를 수행합니다. 자기상관은 효율성과 정확도 면에서 셀프 어텐션를 능가합니다. 장기 예측에서 Autoformer는 에너지, 교통, 경제, 날씨, 질병 등 5가지 실용적 응용 분야를 포괄하는 6개 벤치마크에서 38%의 상대적 개선으로 최첨단 정확도를 달성했습니다.

### chunk 1
- ratio: 1.84
- source_chars: 359
- target_chars: 195

**Source**

The Autoformer model was proposed in Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting by Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long. This model augments the Transformer as a deep decomposition architecture, which can progressively decompose the trend and seasonal components during the forecasting process.

**Target**

The Autoformer 모델은 Haixu Wu, Jiehui Xu, Jianmin Wang, Mingsheng Long가 제안한 오토포머: 장기 시계열 예측을 위한 자기상관 분해 트랜스포머 라는 논문에서 소개 되었습니다. 이 모델은 트랜스포머를 심층 분해 아키텍처로 확장하여, 예측 과정에서 추세와 계절성 요소를 점진적으로 분해할 수 있습니다.
