# Mapped Memory
- CUDA에서 Mapped Memory는 CPU와 GPU 간의 메모리 공유를 가능하게 하여, 두 프로세서가 동일한 메모리 주소를 사용하여 데이터를 접근할 수 있도록 합니다.
- 이를 통해 CPU와 GPU 간의 데이터 전송을 효율적으로 관리합니다.

# cudasetdeviceflags
- CUDA에서 cudaSetDeviceFlags() 함수는 CUDA 장치의 동작 방식을 설정하는 데 사용되며, 동기식 호출(Synchronous Calls)과 관련하여 특정 플래그를 설정할 수 있습니다.
- 이를 통해 CUDA 프로그램의 메모리 관리 및 성능을 최적화합니다.

# Peer-to-Peer Memory
- CUDA에서 Peer-to-Peer (P2P) 메모리는 서로 다른 GPU 간의 직접적인 메모리 접근을 가능하게 하여 데이터 전송의 효율성을 향상시킵니다.
- 이를 통해 두 GPU가 서로의 메모리에 직접 접근할 수 있으며, CPU의 개입 없이 데이터를 주고받을 수 있습니다.

# Texture Memory
- CUDA에서 텍스처 메모리는 GPU에서 주로 그래픽스와 이미지 처리에 사용되는 특수한 메모리 유형입니다.
- 텍스처 메모리는 고속 캐시를 사용하여 메모리 접근 패턴이 지역적인 경우 성능을 크게 향상시킬 수 있습니다.
- 텍스처 메모리는 1D, 2D, 3D 데이터 구조를 지원하며, 주로 이미지 데이터를 처리하는 데 유용합니다.