# 성능 모범 사례

Mobile and embedded devices have limited computational resources, so it is important to keep your application resource efficient. We have compiled a list of best practices and strategies that you can use to improve your TensorFlow Lite model performance.

## 작업에 가장 적합한 모델 선택

작업에 따라 모델 복잡성과 크기 간에 균형을 맞춰야 합니다. 작업에 높은 정확성이 필요하다면 크고 복잡한 모델이 필요할 수 있습니다. 정밀도가 낮은 작업의 경우 디스크 공간과 메모리를 적게 사용할 뿐만 아니라 일반적으로 더 빠르고 에너지 효율적이기 때문에 더 작은 모델을 사용하는 것이 좋습니다. 예를 들어 아래 그래프는 몇 가지 일반적인 이미지 분류 모델에 대한 정확성과 지연 시간 절충을 보여줍니다.

![Graph of model size vs accuracy](../images/performance/model_size_vs_accuracy.png "모델 크기 대 정확도")

![Graph of accuracy vs latency](../images/performance/accuracy_vs_latency.png "정확도 대 지연")

모바일 기기에 최적화된 모델의 한 가지 예는 모바일 비전 애플리케이션에 최적화된 [MobileNet](https://arxiv.org/abs/1704.04861)입니다. [호스팅된 모델](../guide/hosted_models.md)에는 모바일 및 임베디드 기기에 특별히 최적화된 몇 가지 다른 모델이 나열되어 있습니다.

You can retrain the listed models on your own dataset by using transfer learning. Check out our transfer learning tutorial for [image classification](/lite/tutorials/model_maker_image_classification) and [object detection](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193).

## Profile your model

Once you have selected a candidate model that is right for your task, it is a good practice to profile and benchmark your model. TensorFlow Lite [benchmarking tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) has a built-in profiler that shows per operator profiling statistics. This can help in understanding performance bottlenecks and which operators dominate the computation time.

또한, [TensrFlow Lite 추적](measurement.md#trace_tensorflow_lite_internals_in_android)을 사용하여 Android 애플리케이션에서 표준 Android 시스템 추적을 사용하여 모델을 프로파일링하고, GUI 기반 프로파일링 도구로 시간별로 연산자 호출을 시각화할 수 있습니다.

## 그래프에서 연산자 프로파일링 및 최적화

특정 연산자가 모델에 자주 나타나고 프로파일링을 기반으로 해당 연산자가 가장 많은 시간을 소비하는 경우 이 연산자를 최적화할 수 있습니다. TensorFlow Lite는 대부분의 연산자를 위해 최적화된 버전을 가지고 있으므로 이러한 상황이 발생하는 경우는 드뭅니다. 그러나 연산자가 실행되는 제약 조건을 알고 있는 경우 사용자 정의 작업의 더 빠른 버전을 작성할 수 있습니다. [사용자 정의 연산자 설명서](../custom_operators.md)를 확인하세요.

## 모델 최적화

Model optimization aims to create smaller models that are generally faster and more energy efficient, so that they can be deployed on mobile devices. TensorFlow Lite supports multiple optimization techniques, such as quantization.

Check out our [model optimization docs](model_optimization.md) for details.

## 스레드 수 조정

TensorFlow Lite supports multi-threaded kernels for many operators. You can increase the number of threads and speed up execution of operators. Increasing the number of threads will, however, make your model use more resources and power.

For some applications, latency may be more important than energy efficiency. You can increase the number of threads by setting the number of interpreter [threads](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346). Multi-threaded execution, however, comes at the cost of increased performance variability depending on what else is executed concurrently. This is particularly the case for mobile apps. For example, isolated tests may show 2x speed-up vs single-threaded, but, if another app is executing at the same time, it may result in worse performance than single-threaded.

## 중복 사본 제거

애플리케이션이 신중하게 설계되지 않은 경우 모델에 입력을 공급하고 모델에서 출력을 읽을 때 중복 사본이 있을 수 있습니다. 중복된 사본은 제거하십시오. Java와 같은 더 높은 수준의 API를 사용하는 경우 설명서에서 성능 주의사항을 확인하세요. 예를 들어 Java API는 `ByteBuffers`가 [입력](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175)으로 사용되는 경우 훨씬 빠릅니다.

## 플랫폼별 도구로 애플리케이션 프로파일링

Platform specific tools like [Android profiler](https://developer.android.com/studio/profile/android-profiler) and [Instruments](https://help.apple.com/instruments/mac/current/) provide a wealth of profiling information that can be used to debug your app. Sometimes the performance bug may be not in the model but in parts of application code that interact with the model. Make sure to familiarize yourself with platform specific profiling tools and best practices for your platform.

## 기기에서 사용 가능한 하드웨어 가속기로 모델이 얻는 이점 평가

TensorFlow Lite has added new ways to accelerate models with faster hardware like GPUs, DSPs, and neural accelerators. Typically, these accelerators are exposed through [delegate](delegates.md) submodules that take over parts of the interpreter execution. TensorFlow Lite can use delegates by:

- Android의 [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/) 사용하기. 하드웨어 가속기 백엔드를 활용하여 모델의 속도와 효율성을 개선할 수 있습니다. Neural Networks API를 활성화하려면 [NNAPI 대리자](nnapi.md) 가이드를 확인하세요.
- GPU 대리자는 각각 OpenGL/OpenCL 및 Metal을 사용하여 Android 및 iOS에서 사용할 수 있습니다. 사용해 보려면 [GPU 대리자 튜토리얼](gpu.md) 및 [설명서](gpu_advanced.md)를 참조하세요.
- Hexagon 대리자는 Android에서 사용할 수 있습니다. 기기에서 사용할 수 있는 경우 Qualcomm Hexagon DSP를 활용합니다. 자세한 내용은 [Hexagon 대리자 튜토리얼](hexagon_delegate.md)을 참조하세요.
- It is possible to create your own delegate if you have access to non-standard hardware. See [TensorFlow Lite delegates](delegates.md) for more information.

일부 가속기는 다른 유형의 모델에 대해 더 잘 동작합니다. 일부 대리자는 특정 방식으로 최적화된 부동 모델 또는 모델만 지원합니다. 각 대리자를 [벤치마킹](measurement.md)하여 애플리케이션에 적합한지 확인하는 것이 중요합니다. 예를 들어 모델이 매우 작은 경우 모델을 NN API 또는 GPU에 위임하기에 적합하지 않습니다. 반대로, 가속기는 산술 강도가 높은 대형 모델에 사용하기 적합합니다.

## 추가적인 도움이 필요한 경우

TensorFlow 팀은 직면할 수 있는 특정 성능 문제를 진단하고 해결하도록 기꺼이 도움을 드립니다. [GitHub](https://github.com/tensorflow/tensorflow/issues)에 대한 문제를 세부 정보와 함께 제출해주십시오.
