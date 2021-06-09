# TensorFlow Lite 대리자

## 소개

**Delegates** enable hardware acceleration of TensorFlow Lite models by leveraging on-device accelerators such as the GPU and [Digital Signal Processor (DSP)](https://en.wikipedia.org/wiki/Digital_signal_processor).

By default, TensorFlow Lite utilizes CPU kernels that are optimized for the [ARM Neon](https://developer.arm.com/documentation/dht0002/a/Introducing-NEON/NEON-architecture-overview/NEON-instructions) instruction set. However, the CPU is a multi-purpose processor that isn't necessarily optimized for the heavy arithmetic typically found in Machine Learning models (for example, the matrix math involved in convolution and dense layers).

반면에 대부분의 최신 휴대폰에는 이러한 무거운 연산을 더 잘 처리하는 칩이 포함되어 있습니다. 신경망 연산을 위해 활용하면 대기 시간 및 전력 효율성 측면에서 큰 이점이 있습니다. 예를 들어 GPU는 대기 시간을 [최대 5배](https://blog.tensorflow.org/2020/08/faster-mobile-gpu-inference-with-opencl.html)[까지 높일 수 있는 반면 Qualcomm® Hexagon DSP](https://developer.qualcomm.com/software/hexagon-dsp-sdk/dsp-processor)는 실험에서 전력 소비를 최대 75%까지 줄이는 것으로 나타났습니다.

이러한 각 가속기에는 모바일 GPU용 [OpenCL](https://www.khronos.org/opencl/) 또는 [OpenGL ES](https://www.khronos.org/opengles/) 및 DSP용 [Qualcomm® Hexagon SDK](https://developer.qualcomm.com/software/hexagon-dsp-sdk)와 같은 사용자 정의 계산을 가능하게 하는 관련 API가 있습니다. 일반적으로 이러한 인터페이스를 통해 신경망을 실행하려면 많은 사용자 정의 코드를 작성해야 합니다. 각 가속기에 장단점이 있고 신경망에서 모든 작업을 실행할 수 없다는 점을 고려하면 상황이 더욱 복잡해집니다. TensorFlow Lite의 Delegate API는 TFLite 런타임과 이러한 하위 수준 API를 연결하는 다리 역할을 하여 이 문제를 해결합니다.

![Original graph](../images/performance/tflite_delegate_graph_1.png "원본 그래프")

## 대리자 선택

TensorFlow Lite supports multiple delegates, each of which is optimized for certain platform(s) and particular types of models. Usually, there will be multiple delegates applicable to your use-case, depending on two major criteria: the *Platform* (Android or iOS?) you target, and the *Model-type* (floating-point or quantized?) that you are trying to accelerate.

### 플랫폼별 대리자

#### Cross-platform (Android &amp; iOS)

- **GPU delegate** - The GPU delegate can be used on both Android and iOS. It is optimized to run 32-bit and 16-bit float based models where a GPU is available. It also supports 8-bit quantized models and provides GPU performance on par with their float versions. For details on the GPU delegate, see [TensorFlow Lite on GPU](gpu_advanced.md). For step-by-step tutorials on using the GPU delegate with Android and iOS, see [TensorFlow Lite GPU Delegate Tutorial](gpu.md).

#### Android

- **최신 Android 기기용 NNAPI 대리자** - NNAPI 대리자를 사용하여 GPU, DSP 및/또는 NPU를 사용할 수 있는 Android 기기에서 모델을 가속화할 수 있습니다. Android 8.1(API 27+) 이상에서 사용할 수 있습니다. NNAPI 대리자 개요, 단계별 지침 및 모범 사례는 [TensorFlow Lite NNAPI 대리자](nnapi.md)를 참조하세요.
- **구형 Android 기기용 Hexagon 대리자** - Qualcomm Hexagon DSP를 사용하는 Android 기기에서 Hexagon 대리자를 사용하여 모델을 가속화할 수 있습니다. NNAPI를 지원하지 않는 이전 버전의 Android 기기에서 사용할 수 있습니다. 자세한 내용은 [TensorFlow Lite Hexagon 대리자](hexagon_delegate.md)를 참조하세요.

#### iOS

- **최신 iPhone 및 iPad용 Core ML 대리자** - Neural Engine을 사용할 수 있는 최신 iPhone 및 iPad의 경우 Core ML 대리자를 사용하여 32bit 또는 16bit 부동점 모델에 대한 추론을 가속화할 수 있습니다. Neural Engine은 A12 SoC 이상의 Apple 모바일 기기를 사용할 수 있습니다. Core ML 대리자에 대한 개요 및 단계별 지침은 [TensorFlow Lite Core ML 대리자](coreml_delegate.md)를 참조하세요.

### 모델 유형별 대리자

Each accelerator is designed with a certain bit-width of data in mind. If you provide a floating-point model to a delegate that only supports 8-bit quantized operations (such as the [Hexagon delegate](hexagon_delegate.md)), it will reject all its operations and the model will run entirely on the CPU. To avoid such surprises, the table below provides an overview of delegate support based on model type:

**모델 유형** | **GPU** | **NNAPI** | **Hexagon** | **CoreML**
--- | --- | --- | --- | ---
부동점 (32bit) | Yes | Yes | 아니요 | Yes
[훈련 후 float16 양자화](post_training_float16_quant.ipynb) | Yes | 아니요 | 아니요 | Yes
[훈련 후 동적 범위 양자화](post_training_quant.ipynb) | Yes | Yes | 아니요 | 아니요
[훈련 후 정수 양자화](post_training_integer_quant.ipynb) | Yes | Yes | Yes | 아니요
[양자화 인식 훈련](http://www.tensorflow.org/model_optimization/guide/quantization/training) | Yes | Yes | Yes | 아니요

### 성능 검증

The information in this section acts as a rough guideline for shortlisting the delegates that could improve your application. However, it is important to note that each delegate has a pre-defined set of operations it supports, and may perform differently depending on the model and device; for example, the [NNAPI delegate](nnapi.md) may choose to use Google's Edge-TPU on a Pixel phone while utilizing a DSP on another device. Therefore, it is usually recommended that you perform some benchmarking to gauge how useful a delegate is for your needs. This also helps justify the binary size increase associated with attaching a delegate to the TensorFlow Lite runtime.

TensorFlow Lite는 개발자가 자신의 애플리케이션에서 대리자를 사용하는 데 확신을 줄 수 있는 광범위한 성능 및 정확도 평가 도구를 갖추고 있습니다. 이들 도구는 다음 섹션에서 다룹니다.

## 평가 도구

### 지연 시간 및 메모리 공간

TensorFlow Lite의 [벤치마크 도구](https://www.tensorflow.org/lite/performance/measurement)는 적절한 매개변수와 함께 사용하여 평균 추론 지연 시간, 초기화 오버헤드, 메모리 공간 등을 포함한 모델 성능을 추정할 수 있습니다. 이 도구는 모델에 가장 적합한 대리자 구성을 파악하기 위해 여러 플래그를 지원합니다. 예를 들어 `--gpu_backend=gl`를 `--use_gpu`와 함께 지정하여 OpenGL로 GPU 실행을 측정할 수 있습니다. 지원되는 대리자 매개변수의 전체 목록은 [상세한 설명서](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)에 정의되어 있습니다.

다음은 `adb`를 통해 GPU로 양자화된 모델을 실행한 예입니다.

```
adb shell /data/local/tmp/benchmark_model \
  --graph=/data/local/tmp/mobilenet_v1_224_quant.tflite \
  --use_gpu=true
```

You can download pre-built version of this tool for Android, 64-bit ARM architecture [here](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk) ([more details](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark/android)).

### 정확도 및 수정

Delegates usually perform computations at a different precision than their CPU counterparts. As a result, there is an (usually minor) accuracy tradeoff associated with utilizing a delegate for hardware acceleration. Note that this isn't *always* true; for example, since the GPU uses floating-point precision to run quantized models, there might be a slight precision improvement (for e.g., &lt;1% Top-5 improvement in ILSVRC image classification).

TensorFlow Lite에는 지정된 모델에 대해 대리자가 얼마나 정확하게 동작하는지 측정하는 두 가지 유형의 도구, 즉 *Task-Based*와 *Task-Agnostic*이 있습니다. 이 섹션에 설명된 모든 도구는 이전 섹션의 벤치마킹 도구에서 사용한 [고급 델리게이션 매개변수](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)를 지원합니다. 아래의 하위 섹션은 모델 평가(모델 자체가 작업에 적합합니까?)보다 *대리자 평가* (대리자가 CPU와 동일한 작업을 수행합니까?)에 중점을 둡니다.

#### 작업 기반 평가

TensorFlow Lite에는 두 개의 이미지 기반 작업의 정확성을 평가하는 도구가 있습니다.

- [top-K 정확도](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision_at_K)의 [ILSVRC 2012](http://image-net.org/challenges/LSVRC/2012/) (이미지 분류)

- [mean 평균 정밀도 (mAP)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)를 사용하는 [COCO 객체 감지 (경계 상자 포함)](https://cocodataset.org/#detection-2020)

이들 도구(Android, 64bit ARM 아키텍처)의 미리 빌드된 바이너리와 설명서는 여기에서 찾을 수 있습니다.

- [ImageNet 이미지 분류](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_imagenet_image_classification) ([상세 정보](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification))
- [COCO 객체 감지](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_coco_object_detection) ([상세 정보](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/coco_object_detection))

아래 예는 Pixel 4에서 Google의 Edge-TPU를 활용하는 NNAPI를 통한 [이미지 분류 평가](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/imagenet_image_classification)를 보여줍니다.

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/mobilenet_quant_v1_224.tflite \
  --ground_truth_images_path=/data/local/tmp/ilsvrc_images \
  --ground_truth_labels=/data/local/tmp/ilsvrc_validation_labels.txt \
  --model_output_labels=/data/local/tmp/model_output_labels.txt \
  --output_file_path=/data/local/tmp/accuracy_output.txt \
  --num_images=0 # Run on all images. \
  --use_nnapi=true \
  --nnapi_accelerator_name=google-edgetpu
```

예상되는 출력은 1에서 10까지의 Top-K 메트릭 목록입니다.

```
Top-1 Accuracy: 0.733333
Top-2 Accuracy: 0.826667
Top-3 Accuracy: 0.856667
Top-4 Accuracy: 0.87
Top-5 Accuracy: 0.89
Top-6 Accuracy: 0.903333
Top-7 Accuracy: 0.906667
Top-8 Accuracy: 0.913333
Top-9 Accuracy: 0.92
Top-10 Accuracy: 0.923333
```

#### 작업 불가지론적 평가

설정된 온 디바이스 평가 도구가 없는 작업 또는 사용자 정의 모델을 실험하는 경우 TensorFlow Lite에는 [Inference Diff](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/evaluation/tasks/inference_diff) 도구가 있습니다. (Android, 64bit ARM 바이너리 아키텍처 바이너리에 대해서는 [여기](https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_eval_inference_diff)를 클릭)

Inference Diff는 다음 두 가지 설정에서 TensorFlow Lite 실행 (대기 시간 및 출력 값 편차)을 비교합니다.

- 단일 쓰레드 CPU 추론
- 사용자 정의 추론 - [이들 매개변수](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/README.md#tflite-delegate-registrar)에 의해 정의

이를 위해 이 도구는 임의의 가우스 데이터를 생성하고 두 개의 TFLite 인터프리터를 통해 전달합니다. 하나는 단일 스레드 CPU 커널을 실행하고 다른 하나는 사용자 인수에 의해 매개변수화됩니다.

요소별로 각 인터프리터의 출력 텐서 간의 절대 차이뿐만 아니라 두 지연 시간을 측정합니다.

단일 출력 텐서가 있는 모델의 경우 출력은 다음과 같을 수 있습니다.

```
Num evaluation runs: 50
Reference run latency: avg=84364.2(us), std_dev=12525(us)
Test run latency: avg=7281.64(us), std_dev=2089(us)
OutputDiff[0]: avg_error=1.96277e-05, std_dev=6.95767e-06
```

이는 인덱스 `0`의 출력 텐서의 경우 CPU 출력의 요소가 평균 `1.96e-05`만큼 대리자 출력과 다르다는 것을 의미합니다.

이러한 숫자를 해석하려면 모델과 각 출력 텐서가 의미하는 바에 대한 보다 깊은 지식이 필요합니다. 어떤 종류의 점수나 임베딩을 결정하는 단순 회귀인 경우 차이가 낮아야 합니다 (그렇지 않으면 대리자 오류입니다). 그러나 SSD 모델의 '감지 클래스'와 같은 출력은 해석하기가 조금 더 어렵습니다. 예를 들어, 이 도구를 사용하면 차이가 표시될 수 있지만 대리자에게 실제로 문제가 있는 것은 아닙니다. "TV (ID : 10)", "모니터 (ID : 20)"- 대리자는 황금 진실에서 약간 벗어나 TV 대신 모니터를 표시합니다. 이 텐서의 출력 차이는 20-10 = 10만큼 높을 수 있습니다.
