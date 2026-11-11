# 부모 커널 + 25스레드 SHAKE 실행 경로

## 구현 범위

기본 `CDP=1`에서 CPU는 KeyGen/Sign/Verify 부모를 `<<<1,1>>>`로 실행한다.
부모는 device stream을 생성하고 자식들을 순서대로 등록한다. CPU stream을
자식에게 전달하지 않는다. CDP GPU 실행 파일은 병렬 SHAKE만 링크하며,
`cuda/fips202.cu`는 기존 GPU 비교 빌드에서 사용한다. CPU 정답 벡터 생성기는
별도 실행 파일로 빌드한다.
`CDP=0`으로 기존 실행 경로를 선택할 수 있다.

- `parent.cuh`: 실행 순서, CUDA 오류 전달, Sign tail continuation.
- `hashes.cuh`: seed 확장, tr, mu, rhoprime, challenge digest, 최종 비교.
- `sampling.cuh`: ExpandA/ExpandS/ExpandMask/SampleInBall 내부 SHAKE까지 교체.
- `arithmetic.cuh`: 기존 multi의 계산 본문을 재사용한 128스레드 자식들.
- `SHA3.cu`, `SHA3.cuh`: 제공받은 구현. 헤더의 `steam128` 선언 오타만 수정.

제공받은 `symmetric.h`는 기존 파일 위에 덮어쓰지 않았다. 기존 인터페이스는
단일 스레드 기준이고 비교 구현에서 필요하다. 새 경로에서는 `SHA3_CUDA::`
인터페이스를 명시적으로 호출하여 상태 타입과 호출 규약을 혼동하지 않도록 했다.

## 실행 크기와 인자 대응

| 작업 | 실행 크기 | 입력 |
|---|---|---|
| ExpandA | `<<<56,25>>>` | rho 32바이트, nonce `(r<<8)+c`, r=block/7, c=block%7 |
| ExpandS | `<<<15,25>>>` | rhoprime 64바이트, nonce 0..14; 0..6은 s1, 7..14는 s2 |
| ExpandMask | `<<<7,25>>>` | rhoprime 64바이트, `uint16_t(7*nonce+l)` |
| SampleInBall | `<<<1,25>>>` | c_tilde 64바이트; nonce 없음 |
| 단독 SHAKE256 | `<<<1,25>>>` | 아래 표의 입력을 지정 순서로 absorb |
| NTT/행 연산 | 작업 수 × 128스레드 | 다항식 하나당 블록 하나 |

nonce는 기존과 동일하게 하위 바이트, 상위 바이트 순서로 흡수한다.
각 sampling 블록은 독립적인 공유 상태 및 바이트 버퍼를 가진다.
모든 25스레드가 SHAKE의 init/absorb/finalize/squeeze에 함께 참여한다.
rejection/challenge 계수 선택과 gamma1 unpack은 기존 순서를 유지하도록
해당 25스레드 블록의 0번 스레드가 처리한다. 이 부분을 새로 병렬화하지 않았다.
NTT와 행 계산은 128스레드로 분리되어 있다.

| 해시 | absorb 입력 | 출력 |
|---|---|---|
| KeyGen seed | seed[32] || K[1] || L[1] | rho[32] || rhoprime[64] || key[32] |
| tr | pk[2592] | 64바이트 |
| mu | tr[64] || 0x00 || 0x00 || msg[mlen] | 64바이트 |
| 서명 rhoprime | key[32] || zero[32] || mu[64] | 64바이트 |
| c_tilde | mu[64] || packed_w1[1024] | 64바이트 |

문맥 문자열 없음, 결정론적 rnd=0, 최대 메시지 256바이트는 기존 실행 조건과 같다.

## 추가 출력 및 메모리 수명

ExpandA는 기존과 같이 840바이트를 먼저 squeeze하고 부족하면 168바이트씩
추가한다. 잔여 바이트를 보존하는 기존 공식도 유지한다. ETA는 136바이트씩,
gamma1은 680바이트를 생성하고 앞 640바이트를 unpack한다. Challenge는
첫 8바이트의 부호와 이후 rejection 위치를 보존하며 136바이트씩 추가한다.

추가 squeeze는 같은 sampling 자식 커널의 같은 상태로 수행한다. 새 해시를
초기화하지 않는다. 모든 lane이 공유 count/pos를 동기화 후 읽으므로 하나의
lane만 SHAKE에 들어가지 않는다. 커널이 완료될 때까지 상태 수명이 유지된다.

커널 사이 인자는 cudaMalloc 작업 공간 내부의 포인터뿐이다. 부모의 지역 배열이나
공유 배열을 자식에게 전달하지 않는다. 해시 안에서 쓰는 공유 입력 버퍼는 그
해시 커널 내부에서만 소비한다. 한 작업 공간에 여러 최상위 연산을 동시에
실행하는 것은 지원하지 않으며, 배치 호출은 작업 공간을 분리해야 한다.

## 실행 순서

KeyGen: seed → ExpandA → ExpandS → 비밀 계수 pack → NTT(s1) → 행 계산 → H(pk).

Verify: ExpandA → H(pk),mu → challenge → unpack/검사 → NTT → 행 복원 → 해시/비교.

Sign: mu,rhoprime → ExpandA → 비밀키 unpack/NTT → tail로 첫 시도.
각 시도: ExpandMask → NTT(y) → 행 계산 → H(mu||w1) → challenge → NTT(c)
→ z/행 검사 → tail 판정. 성공하면 packing, 실패하면 다음 시도를 tail 실행한다.
각 제어 커널은 자기 grid에서 stream을 새로 만들고 파괴하며 stream 핸들을
다른 grid에 넘기지 않는다. 부모 본문에서 자식의 미완료 결과를 읽지 않는다.

한 stream의 자식들은 순차 실행된다. ExpandA의 56블록은 서로 독립적이지만
서로 다른 단계의 동시 실행을 전제로 하지 않는다. CUDA가 동시 실행을 하지
않더라도 알고리즘이 진행되어야 한다.

## 빌드와 검사

실행 환경을 갖춘 후:

```sh
make check              # CUDA 12+ CDP2, 기본 sm_87
make bench
make CDP=0 check        # 기존 협력 구현
make CDP=0 sweep
make logical-check     # Node.js만 필요, CUDA 빌드 없음
```

`-rdc=true`, `-lcudadevrt`, CUDA 12+ CDP2가 필요하다. CDP1 강제 옵션은 허용하지
않는다. `ARCH`는 실제 GPU에 맞춰 지정한다. 빌드 디렉터리는 cdp와 legacy로
분리하여 옵션 변경 후 이전 실행 파일을 잘못 재사용하지 않는다.

CDP 경로에서 `--nb`는 적용되지 않는다. 호스트 측 blocks=1은 부모 하나를 뜻하며,
자식 블록 수는 위 표와 같다. 기존 단계별 타이머는 CDP를 측정하지 못하므로
해당 결과와 sweep을 제공하지 않는다. 전체 CUDA event 측정과 Sign 반복 횟수는
기존 측정 경로를 사용한다. 성능 개선을 보장하지 않으며 자식 launch 비용이 추가된다.

각 launch/stream API의 첫 오류는 `cdp_error`에 기록한다. 호스트는 결과를 받기 전
그 값을 확인한다. GPU 비동기 실행 오류는 기존 host synchronization으로 확인한다.

## 이번 검토의 한계

CUDA 빌드와 GPU 실행은 하지 않았다. Node 검사에서는 제공된 소스의 Keccak 상수,
rotation, permutation 표를 읽어 25-lane 수학 모델을 만들고 Node/OpenSSL SHAKE와
비교한다. split absorb/squeeze, rate 경계, nonce 및 56개 행렬 매핑, NTT index
대응을 검사한다. 이는 실제 CUDA 코드를 실행한 결과가 아니다.

CUDA 컴파일/링크, GPU 메모리 및 동기화 동작, 전체 ML-DSA 정답 벡터,
Sign 다중 재시도와 compute-sanitizer 검증, 실제 성능은 GPU 환경에서 남아 있다.
