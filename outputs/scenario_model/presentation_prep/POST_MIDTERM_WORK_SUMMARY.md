# 중간발표 이후 진행 내용 전체 정리

이 문서는 **중간발표 이후 우리 팀이 실제로 무엇을 했는지, 왜 그렇게 했는지, 그리고 최종적으로 어떤 인사이트를 얻었는지**를 한 번에 설명하는 상세 참고 문서다.  
최종 발표 슬라이드나 짧은 대본과는 목적이 다르다. 이 문서는 시간 제한 없이 읽는 것을 전제로 하며, **전공자가 아니어도 전체 흐름을 이해할 수 있도록 배경, 맥락, 판단 이유를 최대한 풀어서 설명**하는 것을 목표로 한다.

이 문서의 예상 독자는 다음과 같다.

- 발표 담당자: 어떤 내용을 발표에 넣고 뺄지 판단해야 하는 사람
- 팀원: 전체 프로젝트 흐름을 다시 정리해서 이해해야 하는 사람
- 외부 청중 대비용: “왜 이런 결론이 나왔는가?”를 문장 수준에서 설명할 수 있어야 하는 사람

즉 이 문서는 단순한 결과 요약이 아니라,

1. **문제 정의가 어떻게 구체화되었는지**
2. **모델을 어떻게 확정했는지**
3. **왜 시나리오 분석으로 넘어갔는지**
4. **왜 특정 주가 보강 후보로 선정되었는지**
5. **이 결과를 NGL 포트폴리오 관점에서 어떻게 읽어야 하는지**

를 하나의 이야기로 연결해주는 자료다.

---

## 1. 프로젝트 목표를 다시 정리하면 무엇이었나

중간발표 당시까지는 프로젝트의 초점이 상대적으로 **예측 모델 선택**에 가까웠다.  
즉,

- `RF(Random Forest)`가 더 적합한가,
- `XGB(XGBoost)`가 더 적합한가,
- 어떤 feature set이 성능을 높이는가,
- 왜 2024에서 generalization이 깨지는가

와 같은 질문이 중심이었다.

하지만 프로젝트의 원래 비즈니스 시나리오를 다시 보면, 단순히 “예측 정확도가 높은 모델”을 만드는 것만으로는 충분하지 않았다.

프로젝트 시나리오는 기본적으로 다음과 같은 질문에서 출발했다.

- 앞으로 5년 동안 미국 물류 네트워크는 어디에서 수요가 커질까?
- NGL은 어느 주에 새로운 허브를 두거나, 어떤 기존 허브를 강화해야 할까?
- 단순한 평시 예측이 아니라, 예상치 못한 충격이 왔을 때도 버티는 네트워크를 설계할 수 있을까?

즉 중간발표 이후 프로젝트 목표는 자연스럽게 다음 네 단계로 구체화되었다.

1. **일반화 성능이 안정적인 최종 예측 모델을 확정한다.**  
   단순히 validation이 좋은 모델이 아니라, 실제로 미래와 비슷한 test 기간에도 버티는 모델을 찾는 단계다.

2. **그 예측 모델을 baseline으로 삼아, 향후 허브 중요도와 네트워크 구조를 본다.**  
   여기서 baseline은 “아무 충격이 없을 때의 기본 구조”를 의미한다.

3. **그 baseline 위에 여러 shock 시나리오를 얹는다.**  
   자연재해, 내수 경기 악화, 곡물 공급 감소, 무역 충격 등 다양한 상황을 일부러 만들어서, 네트워크가 어떻게 흔들리는지 본다.

4. **shock를 겪고도 반복적으로 살아남는 보강 후보를 찾는다.**  
   이를 통해 “NGL이 실제로 어디를 새로 확보하거나, 기존 포트폴리오 안에서 어디를 우선 강화해야 하는가”를 제안한다.

즉 중간발표 이후부터는 **모델을 만드는 것 자체가 목적이 아니라, 신뢰할 수 있는 모델을 기반으로 전략적 의사결정을 돕는 것**이 중심 과제가 되었다.

---

## 2. 왜 일반화 성능 안정화에 가장 많은 시간을 썼는가

중간발표 이후 가장 큰 기술적 과제는 **모델 일반화 성능을 안정화하는 것**이었다.

여기서 “일반화 성능”이란,  
모델이 이미 본 데이터에서만 잘 맞는 것이 아니라, **나중 연도나 구조가 조금 바뀐 데이터에서도 어느 정도 버티는 능력**을 뜻한다.

이 프로젝트에서는 특히 `2024` 데이터가 중요했다.  
왜냐하면 많은 모델이 `2023`까지는 그럭저럭 맞추는데, `2024`로 가면 오차가 크게 튀는 현상이 반복되었기 때문이다.

이 문제는 단순히 점수가 조금 나쁜 수준이 아니었다.  
이 상태로 시나리오 분석을 진행하면 다음과 같은 문제가 생긴다.

- baseline 자체가 불안정해진다.
- 특정 주가 중요하게 보이는 이유가 실제 구조 때문인지, 모델 오류 때문인지 구분이 어려워진다.
- 시나리오에서 나타나는 변화가 “real insight”가 아니라 “model noise”일 수 있다.

그래서 우리는 먼저 이렇게 판단했다.

> **시나리오 해석을 믿으려면, baseline forecast가 최소한 2023과 2024에서 같이 버텨야 한다.**

즉 중간발표 이후의 첫 번째 큰 과제는,

- validation 점수를 조금 더 높이는 것보다,
- **2024까지 포함한 generalization을 어떻게든 안정화하는 것**

이었다.

---

## 3. 일반화 문제를 어떻게 바라봤는가

### 3.1 broad macro만으로는 부족했다

초기에는 누구나 자연스럽게 떠올릴 수 있는 가설이 있었다.

- 인플레이션 때문 아닐까?
- 실업률 때문 아닐까?
- 기온이나 경기 분위기 때문 아닐까?
- 금리나 소비 둔화처럼 거시경제 변수 때문 아닐까?

이런 변수들은 전체 경제를 설명하는 데는 유용하다.  
하지만 우리 프로젝트에서 맞춰야 하는 것은 **품목별 주 간 물동량 구조**였다.

실험을 반복해보니 broad macro 변수는 방향성 설명에는 약간 도움이 될 수 있어도,
- `Gasoline`
- `Logs`
- `Gravel`
- `Cereal grains`

처럼 특정 품목에서 발생하는 큰 일반화 실패를 해결해주지는 못했다.

즉 “경제가 좋아졌다/나빠졌다”는 수준의 정보만으로는,
**왜 어떤 품목이 어떤 주에서 특히 크게 움직였는지**를 설명하기 어려웠다.

이 지점에서 얻은 첫 번째 교훈은 명확했다.

> **물류 예측에서 실제로 중요한 건 broad macro보다 commodity-specific structure일 수 있다.**

### 3.2 실패군을 품목별로 따로 봐야 했다

중간발표 이후 우리는 전체 RMSE만 보는 방식에서 벗어나,
**“어떤 품목이 유독 일반화에 실패하는가”**를 따로 보기 시작했다.

그 결과 당시 명확한 실패군은 대략 다음과 같았다.

- `Logs`
- `Gasoline`
- `Gravel`
- `Cereal grains`

이렇게 품목을 따로 보게 되면, 모델 개선 방향도 바뀐다.

- `Logs`에는 목재 생산/처리 구조가 중요하고
- `Gasoline`에는 실제 연료 사용 및 유통 구조가 중요하고
- `Gravel`에는 건설 전체보다 도로·토목 수요가 중요하고
- `Cereal grains`에는 곡물 생산과 출하 구조가 중요하다

는 식으로, **각 실패군마다 서로 다른 설명 변수가 필요하다**는 결론이 나왔다.

즉 일반화 안정화는 “모델을 하나 더 바꿔보자”가 아니라,
**품목별로 실제 구조를 설명하는 데이터를 찾아 붙이는 작업**이 되었다.

---

## 4. 어떤 종류의 데이터 보강을 시도했는가

중간발표 이후의 핵심 작업은 단순히 feature를 많이 넣는 것이 아니라,  
**각 실패 품목을 설명할 수 있는 더 직접적인 외부 데이터가 있는지 찾고, 실제로 generalization이 좋아지는지 검증하는 과정**이었다.

이 과정에서 시도한 데이터는 크게 다섯 부류로 나눌 수 있다.

### 4.1 broad macro / 경기 지표

예:
- 물가
- 경기
- 실업
- 기온
- 일반적인 경기 분위기

이런 변수는 “전체 경제 환경”을 설명한다.  
쉽게 말해,

- 소비가 줄고 있는가,
- 경제가 뜨거운가 식고 있는가,
- 기후가 전반적으로 이상한가

를 보여주는 데이터다.

우리가 처음 이런 지표를 본 이유는, 2024의 구조 변화가 macro regime change일 수 있다고 생각했기 때문이다.  
하지만 결과적으로 이 변수들은 **너무 넓고 거칠었다.**

전체 경제 분위기는 알려주지만,
왜 `Gravel`만 유독 무너졌는지,
왜 `Gasoline`이 특정 주에서 특히 흔들렸는지 같은 문제를 직접 설명하지는 못했다.

### 4.2 commodity-specific direct supply / use 데이터

예:
- 연료 직접 사용량
- 도로 지출
- 목재 산업 활동
- 곡물 생산량

이런 데이터는 훨씬 더 직접적이다.  
예를 들어, 휘발유 물동량을 설명할 때 “유가”보다 “실제로 각 주에서 연료를 얼마나 사용했는가”가 더 직접적인 신호일 수 있다.

이 부류의 데이터가 실제로 가장 잘 먹혔다.

### 4.3 industry demand proxy

예:
- 건설업 고용
- 레미콘 관련 고용
- 시멘트 관련 고용
- 곡물 가공/도매 관련 고용

이런 데이터는 직접적으로 출하량을 보여주지는 않지만,  
어떤 품목을 소비하는 산업이 활발한지 보여주는 간접 지표다.

문제는 이게 **방향은 맞지만 너무 간접적일 수 있다**는 점이다.

예를 들어 건설업 고용이 늘었다고 해서 gravel 출하가 정확히 얼마나 늘어나는지 바로 알 수는 없다.  
곡물 가공업 고용이 있다고 해서 실제 출하 timing이나 merchant flow를 정확히 알 수 있는 것도 아니다.

즉 이 부류는 “설명은 조금 해주지만, 결정적이지는 않은” 데이터였다.

### 4.4 distribution / logistics proxy

예:
- 창고 고용
- 트럭 운송 고용
- 도매업 고용
- county freight concentration

이 부류는 특히 **same-state short-haul** 문제를 설명하기 위해 시도한 것이다.  
즉 “주 내부에서 왜 짧은 거리 물동량을 잘 설명하지 못하는가”를 해결하려고 local distribution 구조를 반영하려 했던 것이다.

하지만 결과적으로 state-year 수준에서는 이 지표들이 너무 거칠었다.  
주 안에서도 대도시권, 산업지대, 농업지대, 항만 배후지가 다 다르기 때문에,
그런 세밀한 구조를 state-level 평균값 하나로 설명하기 어려웠다.

즉 이 부류는 오히려 프로젝트의 한계를 더 분명하게 보여줬다.

### 4.5 niche direct external data

예:
- FHWA highway spending
- USFS timber harvest
- Grain Stocks
- state imports/exports
- tariff episode

이 부류는 broad proxy가 못 잡는 구조를 더 직접적으로 반영하려는 시도였다.  
즉 “겉보기엔 niche하지만, 실제 물동량 구조에 더 가까운 데이터”를 찾아 붙인 것이다.

이 중 일부는 매우 성공적이었고, 일부는 직관과 달리 효과가 거의 없었다.

---

## 5. 실제로 성능 개선에 기여한 데이터와 그 이유

중간발표 이후 가장 중요한 결론은,  
**성능을 실제로 개선한 데이터는 거의 모두 commodity-specific direct data였고, broad proxy는 대부분 한계가 있었다**는 점이다.

### 5.1 Fuel: FHWA direct fuel use

#### 어떤 현실 신호를 넣으려 했나

여기서 넣으려 한 것은 “연료 가격”이 아니라 **실제 주별 연료 사용량**이었다.

쉽게 말하면,
- 연료가 비싼가 싼가보다
- 어느 주가 실제로 연료를 많이 쓰는가

가 휘발유 물동량에 더 직접적일 수 있다고 본 것이다.

#### 왜 이게 중요했나

`Gasoline`은 중간발표 이후에도 큰 실패군이었다.  
그런데 단순한 경기 변수나 가격 변수는 유통 구조를 잘 설명하지 못했다.

연료 사용량은 적어도 “이 주는 실제로 연료를 많이 소비하는가”라는 질문에 직접 답한다.  
따라서 연료 flow를 설명하는 데 더 유리할 수 있다.

#### 왜 어느 정도 먹혔나

실제로 이 보강은 성능을 개선했다.  
완전히 해결한 것은 아니지만, 적어도 “이 방향이 맞다”는 근거를 만들어줬다.

즉 `Fuel` 계열은 broad energy signal보다  
**실제 use data**가 더 유효하다는 걸 확인했다.

### 5.2 Logs: QCEW logging + sawmills

#### 어떤 현실 신호를 넣으려 했나

목재 품목에 대해 제조업 전체를 보는 대신,
**벌목(logging)**과 **제재소(sawmills)** 활동에 집중했다.

#### 왜 이게 중요했나

`Logs`는 일반적인 제조업 경기와는 다르게,
- 실제로 나무를 베는 활동
- 그 원목을 1차 가공하는 활동

에 더 직접적으로 연결된다.

즉 목재 물동량은 “경기가 좋다/나쁘다”보다
**logging과 sawmill activity가 어느 정도인가**가 더 중요하다고 본 것이다.

#### 왜 부분적으로 먹혔나

이 보강은 `Logs` generalization을 일부 개선했다.  
하지만 완전히 해결되진 않았다.

그 이유도 해석할 수 있었다.

- logging / sawmills는 산업활동 자체는 설명한다.
- 하지만 **실제 roundwood receipts나 movement**까지는 설명하지 못한다.

즉 “활동이 있다”는 건 알지만,
“어디서 어디로 얼마나 움직였는가”까지는 이 데이터 하나로는 부족했다.

### 5.3 Gravel: FHWA SF-4 highway spending

#### 어떤 현실 신호를 넣으려 했나

여기서는 broad construction 대신 **실제 도로 지출과 토목 수요**를 보강했다.

#### 왜 이게 중요했나

`Gravel`은 건축 전반과 연결된다고 생각하기 쉽지만,  
실제로는 road base, crushed stone, aggregates 같은 자재 수요와 더 가깝다.

즉 “건설업 전체가 크다”보다  
**도로·교량·토목에 실제 돈이 얼마나 들어가고 있는가**가 더 직접적인 설명 변수가 된다.

#### 왜 크게 먹혔나

이 보강은 중간발표 이후 가장 성공적인 사례였다.  
이전에는 `Gravel`이 명확한 실패군이었는데,
이 데이터를 붙인 뒤에는 사실상 실패군에서 내려오는 수준의 개선이 나타났다.

이 사례는 프로젝트 전체에서 매우 중요한 의미가 있다.

> **좋은 데이터는 “유명한 데이터”가 아니라, 그 품목의 물리적 수요 구조와 가장 가까운 데이터다.**

---

## 6. 시도했지만 효과가 제한적이거나 실패한 데이터와 그 이유

성공한 데이터만큼, 실패한 데이터도 중요했다.  
왜냐하면 실패한 데이터는 **어떤 접근이 한계가 있었는지**를 알려주기 때문이다.

### 6.1 broad macro / broad proxy

예:
- 물가
- 일반 경기 지표
- broad construction proxy
- broad freight index

#### 왜 기대했나

2024 구조 변화가 macro environment와 관련 있을 수 있다고 봤기 때문이다.

#### 왜 잘 안 먹혔나

이 프로젝트에서 남은 오차는 전체 경기 흐름보다,  
**특정 품목의 공급·사용·유통 구조 변화**에 더 가까웠다.

따라서 broad macro는 너무 넓고 추상적이었다.

쉽게 말하면,
- “경제가 나빠졌다”는 건 알려주지만
- “왜 `Gravel`이 여기서 무너졌는가”는 잘 못 알려줬다.

### 6.2 QCEW 기반 건설/가공 수요 proxy

예:
- highway construction
- ready-mix concrete
- cement manufacturing
- grain processing 관련 고용

#### 왜 기대했나

이 품목들을 소비하는 산업이 활발하면 물동량도 늘어날 것이라 생각했기 때문이다.

#### 왜 한계가 있었나

이런 변수는 **활동의 존재**는 알려주지만,
- 실제 출하량
- 실제 프로젝트 집행 규모
- 실제 재고와 출하 timing

까지는 설명하지 못했다.

즉 방향성은 맞아도, 결정적인 정보는 부족했다.

### 6.3 USDA/USGS 원천의 단순 재조합

예:
- 생산량, 면적, 수율, 재고를 방향성 변수나 interaction으로 재구성

#### 왜 기대했나

같은 원천 데이터라도 더 영리하게 조합하면 2024 generalization이 좋아질 수 있다고 봤다.

#### 왜 한계가 있었나

validation이 조금 좋아지는 경우는 있었지만,
정작 2024 test에선 다시 악화되는 경우가 많았다.

즉 기존 원천을 재조합한다고 해서  
**새로운 구조 정보가 생기는 것은 아니었다.**

### 6.4 short-haul용 물류/유통 proxy

예:
- truck transportation employment
- warehousing employment
- wholesale trade employment
- county freight concentration

#### 왜 기대했나

same-state short-haul 문제는 “주 내부 분배 구조를 설명 못한다”는 느낌이 강했기 때문이다.

#### 왜 실패했나

state-year 수준에서는 이 지표들이 너무 평균적이었다.  
실제 주 내부 물류는
- 대도시권
- 공업지대
- 농업지대
- 항만 배후지

에 따라 매우 다르다.  
그런데 state-level 고용 비중 하나로 그 차이를 설명하기 어렵다.

이 실패는 오히려 프로젝트의 한계를 선명하게 보여줬다.

> **same-state short-haul 문제는 “좋은 state-level 변수가 없다”기보다, state-level 자체가 너무 거친 해상도일 수 있다.**

### 6.5 niche direct data 중에서도 한계가 있었던 것

#### USFS timber harvest

직관적으로는 `Logs`에 아주 잘 맞을 것처럼 보였다.  
실제로도 데이터 의미는 그럴듯하다.

하지만 성능 개선은 크지 않았다.

이유는, timber harvest가 “얼마나 생산했는가”는 보여줄 수 있어도,
우리가 맞추려는 **실제 movement pattern**과 일대일로 연결되진 않기 때문이다.

#### Grain Stocks

곡물 재고는 공급망 구조에서 분명 중요하다.  
하지만 이번 구성에선 2024 generalization을 크게 바꾸지 못했다.

그 이유는 재고만으로는,
- 언제 출하되는지
- 어느 방향으로 merchant flow가 움직이는지

를 충분히 설명하지 못했기 때문이다.

---

## 7. 이 일반화 안정화 과정에서 얻은 핵심 교훈

이 과정은 단순한 모델 튜닝이 아니었다.  
사실상 **“물류 예측에서 무엇이 진짜 설명력이 있는 데이터인가”**를 가려내는 과정이었다.

여기서 얻은 교훈은 크게 세 가지였다.

### 7.1 broad macro보다 direct operational data가 훨씬 중요했다

실제로 먹힌 건
- 연료 직접 사용량
- 도로 지출
- 목재 관련 실제 산업활동

처럼 물동량 구조와 가까운 데이터였다.

즉 현실에 가까운 operational data가 강했다.

### 7.2 proxy는 방향은 맞아도 결정적이지 않았다

고용, broad industry activity, 일반 경기지표는
- “왜 그럴 수도 있는지”는 설명하지만
- “실제로 얼마나 움직였는지”는 잘 못 잡았다.

즉 좋은 보조 설명은 될 수 있지만, 핵심 해결책은 아니었다.

### 7.3 남은 실패는 모델 복잡도보다 데이터 해상도 문제에 가까웠다

특히 same-state short-haul은
- 더 많은 state-level 변수를 넣는다고 해결될 문제가 아니었고,
- state-level보다 더 세밀한 공간 구조가 필요하다는 한계를 드러냈다.

이 점이 이후 시나리오 해석 범위를 state-level로 제한한 중요한 이유 중 하나다.

즉 최종 모델이 중요한 이유는  
“우연히 점수가 제일 좋았기 때문”이 아니라,
**직접 설명력이 확인된 데이터만 남겨서 generalization을 안정화한 결과물이기 때문**이다.

---

## 8. 끝까지 남은 한계

다음은 끝까지 남은 어려움이었다.

- `Gasoline`
- `Logs`
- `Cereal grains`
- `same-state short-haul`

여기서 중요한 해석은,
- 모델이 단순히 “못 배웠다”기보다,
- 2024에 commodity-specific structure shift가 있었고,
- 이를 설명할 state-level direct data가 한계에 부딪혔다는 점이었다.

이 때문에 최종적으로 시나리오 해석 범위도 **state-level interstate 중심**으로 제한하게 되었다.

---

## 8-1. 왜 최종적으로 RF 계열이 XGB보다 더 안정적으로 보였는가

이 프로젝트를 잘 모르는 사람 입장에서는 자연스럽게 이런 질문이 생길 수 있다.

> “보통은 gradient boosting 계열이 더 강력한 모델 아닌가?  
> 그런데 왜 이번엔 RF 쪽이 더 안정적으로 보였는가?”

이 질문은 충분히 타당하다.  
실제로 이 프로젝트에서도 초반에는 `XGB`가 충분히 유력한 challenger였다.  
하지만 최종적으로는 `RF` 계열이 더 안정적인 baseline으로 남았다.

이걸 이해하려면 먼저 두 모델이 데이터를 다루는 방식 차이를 간단히 볼 필요가 있다.

### 8-1.1 RF와 XGB의 차이

#### Random Forest

`RF`는 많은 결정트리를 서로 다르게 학습시킨 뒤,  
그 결과를 **평균**내는 방식이다.

이 구조의 장점은:
- 한두 개의 복잡한 패턴에 과하게 끌리지 않고
- 전반적으로 **보수적이고 안정적인 예측**을 만든다는 점이다.

쉽게 말하면 RF는
“정확히 하나의 날카로운 답”보다는  
“여러 나무의 평균적인 답”을 내는 방식에 가깝다.

#### XGBoost

`XGB`는 이전 단계에서 남은 오차, 즉 **residual**을 계속 줄여가며  
다음 트리를 순차적으로 쌓는 방식이다.

이 구조의 장점은:
- 복잡한 interaction을 잘 잡고
- 훈련 데이터에서 남는 오차를 매우 적극적으로 줄인다는 점이다.

반대로 말하면 XGB는
“이상한 패턴이든 미세한 구조든, 남은 오차를 설명할 수 있으면 끝까지 따라가려는 모델”
라고 볼 수 있다.

이번 프로젝트에서 중요한 건,  
**이 ‘적극성’이 항상 일반화에 유리하진 않았다는 점**이다.

### 8-1.2 이 프로젝트 구조가 왜 RF에 유리했을 가능성이 큰가

이번 문제의 입력은 기본적으로
- `state-level`
- `year-level`
- `commodity`, `dist_band`, `trade_type`
조합이었다.

그런데 실제 물류는 특히 단거리에서
- 주 내부 분배
- 도시권별 창고 구조
- metro 간 재분배
- local trucking

같은 훨씬 더 세밀한 구조로 움직인다.

즉,
- **입력은 거칠고**
- **실제 타깃은 더 복잡한 구조를 품고 있는**
문제였다.

이런 상황에서 XGB는 남은 오차를 줄이기 위해,
실제로는 관측되지 않은 구조를 다른 feature 조합으로 “억지로 설명”하려 들 수 있다.

반면 RF는 평균화 때문에
- 극단적인 residual 패턴을 덜 따라가고
- 덜 화려하지만 더 안정적인 함수를 만든다.

이 점이 2024 generalization에서 RF에 유리하게 작용했을 가능성이 크다.

### 8-1.3 same-state short-haul이 왜 중요한 힌트였나

이 프로젝트에서 특히 중요한 단서는
**단거리 문제의 대부분이 same-state short-haul이었다**는 점이다.

우리가 later-stage analysis에서 확인한 바에 따르면,
단거리 오차의 본체는 거의
- `same-state`
- local distribution
- broad state-level 변수로는 잘 설명되지 않는 구조

였다.

이게 왜 중요하냐면,
same-state short-haul은 실제로는 sub-state 정보가 필요할 가능성이 큰 영역이기 때문이다.

예를 들어 같은 텍사스 주 안에서도
- 대도시권
- 항만 배후지
- 공업지대
- 농업지대

가 다르다.  
그런데 우리는 이걸 state-level 평균 변수로만 보고 있었다.

이럴 때 XGB는
- 주 내부의 미묘한 잔차를
- 다른 변수 조합으로 설명하려고 하다가
- train/validation에 맞춘 세밀한 규칙을 만들 수 있다.

문제는 그 규칙이 2024처럼 구조가 조금만 바뀌어도 깨지기 쉽다는 것이다.

반면 RF는
- 그런 잔차를 덜 공격적으로 따라가고
- 상대적으로 **덜 과적합된 평균적 예측**을 만든다.

즉,

> **same-state short-haul 비중이 큰 구조가 RF 우세의 중요한 원인 중 하나였을 가능성이 높다.**

라는 해석은 꽤 설득력이 있다.

### 8-1.4 heavy-tail과 sparse flow도 RF에 유리했을 수 있다

또 하나 중요한 점은, 실패군의 분포가 매우 균일하지 않았다는 것이다.

특히
- `Logs`
- `Gasoline`
- `Gravel`
- `Cereal grains`

같은 품목은 일부 대형 흐름이 전체 오차를 크게 좌우하는 **heavy-tail 구조**를 보였다.

이런 데이터에서는 boosting 계열이
- 큰 오차가 나는 일부 패턴을 계속 따라가다가
- 특정 연도와 특정 구조에 맞는 규칙을 만들 가능성이 있다.

그런데 우리가 원하는 건 train에 대한 정밀한 적합이 아니라,
**2024처럼 구조가 달라진 시점에도 버티는 것**이었다.

RF는 이런 상황에서
- tail을 완벽히 맞추진 못하더라도
- 평균적으로 덜 흔들리는 경향이 있다.

즉 이번 문제는 “최고의 fit”보다 **안정성**이 더 중요했고,
그 점에서 RF가 유리했을 가능성이 있다.

### 8-1.5 결국 더 중요했던 건 모델 복잡도보다 데이터 품질이었다

중간발표 이후 실제 성능 개선은
- 더 정교한 booster 튜닝
- 더 복잡한 interaction

보다도,

- 연료 직접 사용량
- 도로 지출
- logging / sawmill activity

같이 **더 직접적인 데이터를 붙였을 때** 나왔다.

이건 매우 중요한 시사점을 준다.

이번 문제는
- 모델이 충분히 강력하지 않아서 실패한 게 아니라,
- **설명 변수가 실제 구조를 얼마나 직접적으로 반영하느냐**가 더 중요한 문제였다는 뜻이다.

이럴 때는 XGB의 강점인
- residual 세밀 추적
- 복잡한 비선형 interaction 학습

이 상대적으로 덜 중요해지고,

RF처럼
- 좋은 feature가 들어왔을 때
- 그 신호를 과하게 비틀지 않고
- 보수적으로 평균내는 모델

이 더 안정적으로 보일 수 있다.

### 8-1.6 이 해석을 어떻게 받아들이는 게 좋은가

이걸 “RF가 원래 XGB보다 우수하다”로 받아들이면 안 된다.  
정확한 해석은 다음에 가깝다.

> 이번 프로젝트처럼  
> `state-level의 거친 입력`,  
> `same-state short-haul의 큰 비중`,  
> `heavy-tail / sparse flow`,  
> `commodity-specific direct data의 중요성`  
> 이 동시에 존재하는 문제에서는,  
> XGB보다 RF가 더 안정적으로 일반화했을 가능성이 크다.

즉 RF 우세는 “모델 클래스의 승패”가 아니라,
**이번 문제 구조에서 나타난 결과**로 이해하는 것이 맞다.

### 8-1.7 최종적으로 남기는 한 줄 요약

이 프로젝트에서 RF가 XGB보다 더 좋은 baseline으로 남은 가장 큰 이유는,

**state-level의 거친 입력으로 same-state short-haul과 commodity-specific 구조 변화를 완전히 설명할 수 없는 상황에서, XGB는 남는 잔차를 더 공격적으로 따라가다가 2024 일반화에서 불리해졌고, RF는 평균화 덕분에 더 안정적으로 버텼기 때문**이라고 해석할 수 있다.

---

## 9. 허브 예측으로의 전환

### 9.1 2031까지의 baseline forecast

최종 모델을 확정한 뒤, 이를 기반으로 **2031까지의 state-level hub forecast**를 생성했다.

여기서 “허브”는 단순히 물동량이 많은 주를 뜻하는 것이 아니다.  
우리는 각 주가
- 얼마나 많은 흐름을 다루는지,
- 여러 주와 얼마나 연결되는지,
- 특정 shock가 왔을 때 얼마나 중요한지

를 함께 보면서 “허브 후보”를 해석했다.

이 단계의 핵심 메시지는:

- 현재 구조를 기준으로 봤을 때
- `TX, CA, IL, FL, PA, NY, OH, IN, MI, TN`
가 지속적으로 강한 허브 후보라는 점이었다.

이 예측은 공격적인 미래 성장 추정보다,
- **현재 구조 기준의 안정적 허브 후보**
를 보는 보수적 baseline 성격이 강했다.

즉 이 단계는 “새로운 surprise candidate를 찾는 것”보다,
**현재 구조에서 이미 강한 주를 baseline으로 확인하는 과정**이었다.

### 9.2 허브의 두 역할

예측 결과를 보면 허브는 단일 개념이 아니었다.

#### Backbone

`Backbone`은 주와 주 사이를 잇는 **interstate connectivity 중심 허브**다.

이런 허브는
- 다른 주와의 연결이 많고
- 네트워크 전체의 통로 역할을 하고
- 하나가 흔들리면 여러 주로 파급될 가능성이 크다.

즉 backbone은 “네트워크 연결의 뼈대”에 가깝다.

#### Internal

`Internal`은 주 내부 분배나 same-state 흐름 비중이 큰 허브다.

이런 허브는
- 큰 내수권을 떠받치고
- local buffering 역할을 하며
- 외부 충격이 왔을 때도 주 내부에서 수요를 흡수할 가능성이 있다.

즉 internal은 “주 내부 분배의 중심축”에 가깝다.

이 구분은 이후 시뮬레이션과 NGL 제안에서 매우 중요했다.  
왜냐하면 기업 입장에서 필요한 건 단순히 “큰 주”가 아니라,
**무슨 역할을 하는 허브인가**이기 때문이다.

---

## 10. 시나리오 설계 원칙

회의를 통해 시나리오 설계 원칙을 확정했다.

### 10.1 해상도

- **state-level 고정**
- sub-state 해석은 이번 범위에서 제외

왜냐하면 현재 모델과 외부 데이터의 대부분이 state-level이기 때문이다.  
이보다 더 미세한 단위로 가면, 모델이 실제로 학습하지 않은 수준의 결론을 억지로 내리게 된다.

### 10.2 shock family

시나리오는 크게 네 family로 묶었다.

1. `Capacity shock`
2. `Demand shock`
3. `Commodity-specific supply shock`
4. `Trade/external network shock`

이렇게 분류한 이유는 “사건 이름”보다 “물류에 어떤 메커니즘으로 작용하느냐”가 더 중요했기 때문이다.

예를 들어,
- 폭설과 산불은 원인은 다르지만 둘 다 capacity shock가 될 수 있고,
- 곡물 흉작은 자연재해 원인을 가질 수 있어도 물류 메커니즘상 supply shock로 보는 게 더 자연스럽다.

### 10.3 주입 방식

1차는 **state blocking**을 기본으로 두고,
필요 시 일부 시나리오에서 **feature perturbation** 또는 demand overlay를 추가하는 방식으로 정리했다.

쉽게 말하면,

- 어떤 shock에서는 특정 주의 flow capacity를 깎고
- 어떤 shock에서는 특정 품목의 공급이나 수요를 약화시키는 방식이다.

### 10.4 shock 강도 규칙

- `mild`: `5%`, 1년 지속
- `medium`: `20%`, 3년 지속
- `severe`: `40%`, 5년 지속
- 1년 초과 shock는 매년 절반씩 감쇄
  - 예: `40 -> 20 -> 10 -> 5 -> 2.5`

이 규칙을 고정한 이유는 시뮬레이션이 임의적이라는 비판을 줄이기 위해서다.  
shock 강도와 지속기간을 먼저 정해둔 뒤, 각 family에 동일한 기준을 적용했다.

### 10.5 제외 범위

- `same-state short-haul`은 시뮬레이션 본편에서 제외

이유는 명확하다.

1. state-level 해석과 잘 안 맞고
2. 주 내부 분배 구조를 설명할 데이터가 부족하며
3. 여기에 시간을 더 쓰면 전체 발표의 핵심이 흐려질 수 있기 때문이다

즉 “중요하지 않아서”가 아니라,  
**현재 프로젝트의 해상도와 시간 범위 안에선 다루기 어렵기 때문에 의식적으로 제외한 것**이다.

### 10.6 왜 항상 등장하는 backbone / core 허브를 제외하고 next candidate를 봤는가

시나리오 해석에서 또 하나 중요한 원칙은,  
**평시에도 늘 상위에 남는 backbone 또는 core 허브를 그대로 포함한 채 후보를 읽지 않았다는 점**이다.

이유는 단순하다.  
`TX`, `CA`, `FL`, `PA`처럼 baseline에서도 이미 매우 크고, shock를 줘도 계속 상위권에 남는 주들이 있다.  
이 주들을 그대로 두면 시나리오별 결과가 매번 비슷해지고,
결국 “원래 큰 주가 또 중요했다”는 말만 반복하게 된다.

하지만 우리 팀이 알고 싶었던 것은 그보다 더 실무적인 질문이었다.

- 주요 backbone이 흔들렸을 때 어떤 주가 보조 허브로 떠오르는가?
- NGL이 이미 가진 핵심 footprint 바깥에서 새로 보강할 가치가 있는 주는 어디인가?
- 기존 거대 허브를 제외하고도 반복적으로 살아남는 후보가 있는가?

그래서 시나리오별 ranking을 볼 때는  
**항상 상위에 남는 backbone/core 허브를 먼저 제외한 뒤, 그 다음에 나타나는 next candidate를 따로 봤다.**

이 원칙 덕분에 시나리오 분석은  
“누가 원래 큰가”를 다시 확인하는 작업이 아니라,  
**“충격이 왔을 때 어떤 보강 후보가 추가로 의미를 갖는가”**를 찾는 작업이 될 수 있었다.

이후 `must-have exclusion`, `backbone-only exclusion`, `Top 10/Top 15 next candidate` 같은 세부 실험도 모두 이 문제의식에서 나온 것이다.

---

## 11. 자연재해 트랙

자연재해는 세 갈래로 나누어 진행했다.

- `snow / heat`
- `tornado`
- `wildfire / smoke`

이 트랙의 핵심은 **broad network stress**였다.  
즉 자연재해는 특정 한 품목만 흔드는 것이 아니라, 주 전체의 이동 capacity를 광범위하게 깎을 수 있다는 가정이다.

요약 결과:
- `snow_heat`: cumulative network loss `1.85%`
- `tornado`: cumulative network loss `1.64%`
- `wildfire_smoke`: cumulative network loss `1.46%`

### 왜 중요한가

이 숫자는 절대값만 보면 크지 않아 보일 수 있다.  
하지만 다른 family들과 비교해보면, 자연재해는 **전체 네트워크에 가장 넓게 영향을 미치는 broad stress**였다.

즉 곡물 shock처럼 특정 commodity만 흔드는 게 아니라,
여러 품목과 여러 주에 동시에 영향을 준다.

### 어떤 인사이트가 나왔나

이 트랙에서 특히 흥미로웠던 것은 `AR`이었다.

`AR`은 평시 기준으론 절대적인 core hub라고 보기 어렵다.  
그런데 자연재해 시나리오에선 `TX-LA-OK-MS` 축이 흔들릴 때 살아남는 **south-central fallback hub**처럼 보였다.

즉 자연재해 트랙은
- “평소 누가 큰가”보다
- **주요 축이 흔들릴 때 누가 보조축이 되는가**

를 잘 보여주는 family였다.

관련 문서:
- [STRATEGIC_INSIGHTS_AR_AND_NATURAL_DISASTER.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/STRATEGIC_INSIGHTS_AR_AND_NATURAL_DISASTER.md)

---

## 12. 내수 문제 트랙

내수 문제는 하나로 묶지 않고 세 갈래로 분리했다.

1. `construction / infrastructure hybrid`
2. `domestic bottleneck / congestion`
3. `domestic broad demand`

이렇게 나눈 이유는 모두 “내수 문제”처럼 보여도,
네트워크에 작용하는 방식이 서로 다르기 때문이다.

### 12.1 construction / infrastructure

- cumulative network loss `2.33%`
- material subset cumulative loss `8.62%`

이건 **건설자재 수요 둔화형 shock**다.

즉 도로 지출과 건설 활동이 약해지면,
- `Gravel`
- `Nonmetal mineral products`
- aggregates 계열

같은 자재의 움직임이 줄어드는 상황을 보는 것이다.

### 12.2 bottleneck / congestion

- cumulative network loss `2.80%`
- interstate cumulative loss `13.41%`

이건 **interstate reliability deterioration**다.

즉 소비가 줄어서가 아니라,
- 혼잡이 심해지고
- 트럭 이동 신뢰도가 떨어지고
- 주요 corridor가 병목을 일으키는 상황

을 보는 것이다.

따라서 construction과 bottleneck은 둘 다 “내수 문제”이지만,
- 하나는 수요 둔화형
- 다른 하나는 연결성 악화형

이다.

### 12.3 broad demand

- cumulative network loss `0.15%`
- interstate cumulative loss `0.74%`

이건 상대적으로 완만한 **consumer-demand slowdown**이다.

쉽게 말해,
- 플로리다, 텍사스, 캘리포니아 같은 큰 소비 주의 수요가 약해질 때
- 어떤 주가 그래도 support candidate로 남는가

를 보는 시나리오다.

### 12.4 왜 `NC`와 `VA`가 중요했나

이 세 갈래를 같이 놓고 봤을 때,
`NC`와 `VA`는 construction, bottleneck, broad demand에서 반복적으로 살아남았다.

이건 매우 중요한 신호다.  
즉 `NC`, `VA`는 특정 한 종류의 domestic shock에서만 뜨는 예외가 아니라,
**동부/동남부를 지지하는 비교적 안정적인 보강 축**일 가능성이 높다는 뜻이다.

관련 문서:
- [STRATEGIC_INSIGHTS_NC_VA_DOMESTIC_SHOCKS.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/STRATEGIC_INSIGHTS_NC_VA_DOMESTIC_SHOCKS.md)

---

## 13. Grain supply 트랙

풍작/흉작은 기후와 연결돼 있지만, 시뮬레이션상으로는 **commodity-specific supply shock**로 분류했다.

이건 중요한 선택이었다.  
왜냐하면 곡물 수확량 변화는 자연재해 원인을 가질 수 있어도, 물류적으로는 결국 **공급량 변화**로 나타나기 때문이다.

선정 주:
- `IA` severe
- `TX`, `NE` medium
- `IL`, `MN` mild

결과:
- cumulative network loss `0.19%`
- cereal-only cumulative loss `2.42%`
- cereal-only peak loss `9.91%`

### 어떻게 읽어야 하나

이 shock는 전체 네트워크를 뒤흔드는 broad shock는 아니다.  
대신 **곡물 계열 물동량을 강하게 흔드는 narrow shock**다.

즉 이 family의 핵심은 “전체망 붕괴”가 아니라,
**곡물 공급망이 흔들릴 때 누가 중심을 대신 받쳐주는가**다.

### 핵심 보강 후보

- `MN`, `NE`, `KS`: surviving grain core
- `WA`, `OR`: Pacific Northwest export support axis

해석하면,
- `MN`, `NE`, `KS`는 곡물 belt의 중심축을 이어받는 내륙 core이고
- `WA`, `OR`는 곡물이 최종적으로 빠져나가는 export support 역할을 한다

는 뜻이다.

관련 문서:
- [STRATEGIC_INSIGHTS_GRAIN_REALWORLD_INTERPRETATION.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/STRATEGIC_INSIGHTS_GRAIN_REALWORLD_INTERPRETATION.md)

---

## 14. Trade / external network 트랙

### 14.1 왜 1차 버전은 부족했나

초기 trade shock는 사실상
- Census state exports
- China share
- vessel/container exposure

중심이었다.

즉 “어느 주가 수출과 해상 운송에 많이 노출돼 있나”를 보는 정도였다.  
이건 trade shock의 일부는 설명하지만, 충분하지 않았다.

왜냐하면 실제 무역 충격은
- 수출만이 아니라 수입도 중요하고
- 컨테이너/항만 의존도도 중요하고
- 특정 국가와의 trade-war exposure도 중요하기 때문이다.

### 14.2 enhanced trade 버전

그래서 이후 다음 데이터를 추가했다.

- Census state exports
- Census state imports
- USTR Section 301 China
- USTR Section 232 Canada/Mexico

즉 shock 정의를
- `exports only`
에서
- `imports + containers + partner-country dependence`

로 확장했다.

선정 주:
- `CA` severe
- `GA`, `NJ` medium
- `MI`, `IL` mild

결과:
- cumulative network loss `0.32%`
- interstate cumulative loss `1.54%`

### 어떻게 읽어야 하나

이 family는 자연재해처럼 전체를 넓게 흔드는 건 아니지만,
**항만·수입·컨테이너·국가별 trade exposure가 겹친 주들의 취약성**을 잘 보여준다.

핵심 해석:
- `NJ`, `GA`: container-heavy gateway reinforcement
- `NC`, `VA`: East Coast inland connector reinforcement
- `UT`: secondary inland fallback

즉 trade family에서 `NC`, `VA`가 다시 등장한 것은,
이 둘이 단순 소비 주가 아니라 **동부 corridor와 inland distribution을 같이 받쳐주는 주**이기 때문이다.

관련 공식 문서:
- [TRADE_EXTERNAL_NETWORK_SHOCK_REFERENCE.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_enhanced_shock/TRADE_EXTERNAL_NETWORK_SHOCK_REFERENCE.md)

---

## 15. 교차 비교

모든 family를 교차 비교한 결과, 반복적으로 등장하는 후보와 shock-specific 후보가 나뉘었다.

가장 자주 살아남은 주:
- `WA`, `MN`, `NC`, `NE`, `CO`, `KS`

공통 scenario-only 후보:
- `NC`
- `VA`
- `UT`
- `SC`
- `ND`
- `NM`
- `NJ`

### 이게 왜 중요한가

각 family를 따로 보면 특정 주가 우연히 좋아 보일 수 있다.  
하지만 여러 family를 가로질러 반복해서 남는다는 것은,
그 주가 **특정 이벤트 한 번의 결과가 아니라 구조적으로 유용한 주**일 가능성을 높여준다.

여기서 가장 강한 메시지는 역시 `NC`, `VA`다.

- `NC`, `VA`는 더 이상 단일 family의 예외가 아니라,
  **여러 shock family에서 반복적으로 살아남는 동부/동남부 보강 후보**가 되었다.

관련 문서:
- [STRATEGIC_INSIGHTS_CROSS_FAMILY.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/cross_family_comparison/STRATEGIC_INSIGHTS_CROSS_FAMILY.md)

---

## 16. 최종 보강 후보 재정의

최종적으로 후보군을 네 층으로 재정의했다.

### 16.1 Core expansion

- `NC`
- `VA`

이 둘은 현재 NGL footprint 바깥에서,  
가장 반복적으로 살아남은 신규 보강 후보다.

### 16.2 Robust internal reinforcement

- `WA`
- `MN`
- `NE`
- `KS`
- `CO`

이 다섯 주는 이미 포트폴리오 안에 있거나,  
현재 구조와 여러 shock family를 동시에 놓고 봤을 때 **반복적으로 강한 내부 보강축**이었다.

### 16.3 Secondary internal reinforcement

- `OR`
- `WI`
- `AL`
- `LA`

이들은 바로 최우선은 아니지만,  
여전히 여러 family에서 의미 있는 2선 후보다.

### 16.4 Conditional watchlist

- `UT`
- `SC`
- `ND`
- `NM`
- `NJ`
- `SD`
- `NV`

이들은 특정 family나 특정 조건에서 의미가 있지만,
모든 경우에 강한 것은 아닌 조건부 후보다.

### 핵심 메시지

- 신규 확장을 묻는다면 `NC`, `VA`
- 기존 포트폴리오 내부 보강을 묻는다면 `WA`, `MN`, `NE`, `KS`, `CO`

관련 문서:
- [FINAL_REINFORCEMENT_CANDIDATE_REFERENCE.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/final_reinforcement_candidates/FINAL_REINFORCEMENT_CANDIDATE_REFERENCE.md)

---

## 17. 품목별 NGL 제안

최종적으로 품목별 제안도 정리했다.

- `Cereal grains`: `NE`, `MN`, `KS`
- `Construction materials / aggregates`: `NC`, `VA`, `CO`, `WI`
- `Logs / wood-linked`: `WA`, `OR`, `AL`, `NC`
- `Trade redistribution / gateway`: `WA`, `LA`, 보조 `NC`, `VA`
- `Fuel / energy balancing`: `KS`, `LA`, 보조 `UT`, `NJ`
- `Trade-specific adjunct`: `NJ`, `GA`, 보조 `SC`, `UT`

이 표의 의미는 단순히 “이 주가 좋다”가 아니다.  
어떤 품목에서는 어떤 주가 더 직접적인 보강 효과를 낼 수 있는지를 정리한 것이다.

즉 NGL 입장에서는 전체 확장 전략뿐 아니라,
**특정 commodity portfolio를 강화하고 싶을 때 어느 주가 유리한가**까지 볼 수 있다.

관련 문서:
- [NGL_COMMODITY_SPECIFIC_RECOMMENDATIONS.md](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/ngl_commodity_recommendations/NGL_COMMODITY_SPECIFIC_RECOMMENDATIONS.md)

---

## 18. 최종적으로 발표에서 강조할 수 있는 메시지

이 문서를 전부 압축해서 발표 메시지로 정리하면 다음 다섯 줄이 핵심이다.

1. **중간발표 이후 가장 큰 진전은 일반화 성능 안정화였다.**  
   단순 알고리즘 비교를 넘어서, direct data를 붙이며 2024까지 버티는 baseline을 확보했다.

2. **최종 모델을 기반으로 허브 예측과 state-level resilience simulation을 연결했다.**  
   즉 예측에서 끝난 것이 아니라, shock 대응 전략까지 확장했다.

3. **NGL은 이미 state-level 기준으로 성숙한 포트폴리오를 갖고 있다.**  
   따라서 무작정 새 주를 많이 추가하는 전략보다, 보강형 전략이 더 중요하다.

4. **가장 설득력 있는 신규 보강 후보는 `NC`와 `VA`다.**  
   이 둘은 여러 family를 가로질러 반복적으로 등장했다.

5. **기존 포트폴리오 내부 보강 후보는 `WA`, `MN`, `NE`, `KS`, `CO`다.**  
   즉 이미 가진 footprint 안에서도 우선순위 재정렬이 필요하다.

---

## 19. 발표 자료에 넣을지 말지 판단해야 하는 내용

### 넣는 것이 좋은 내용

- final model 성능 개선
- 왜 시나리오를 해야 했는지
- shock family 구조
- `NC`, `VA`가 왜 핵심인지
- 최종 후보군과 품목별 제안

### 필요하면 줄여도 되는 내용

- RF vs XGB의 세부 탐색 과정
- 개별 feature 실험의 상세 실패 사례
- 모든 지도 전체
- `Top10/Top15`, `must-have/backbone-only`의 계산 세부

### 백업 슬라이드로 두면 좋은 내용

- same-state short-haul 한계
- trade shock 1차 버전과 enhanced 버전 차이
- `AR`, `NJ`, `UT` 같은 조건부 후보 해석

---

## 20. 이 문서를 읽을 때의 한 줄 요약

이 프로젝트의 본질은  
**“좋은 예측 모델을 골랐다”**가 아니라,
**“일반화가 버티는 모델을 만든 뒤, 그 모델을 기반으로 NGL의 허브 포트폴리오를 shock-resilient하게 재해석했다”**는 데 있다.

---

## Appendix A. 핵심 시각자료 모음

이 부록은 중간발표 이후 우리가 만든 핵심 그래프와 지도를 **주제별로 바로 확인할 수 있게 정리한 시각자료 인덱스**다.  
정적 그림은 문서 안에 바로 붙였고, interactive 지도는 별도 링크를 함께 제공한다.

### A-1. 모델 일반화 안정화

이 세 그래프는 중간발표 이후 가장 중요한 기술적 진전이 **모델 일반화 안정화**였다는 점을 보여준다.

- overall RMSE progress  
  ![overall rmse progress](/C:/Users/서동준/IdeaProjects/FAF5.7.1_2018-2024/outputs/scenario_model/model_progress_visuals/01_overall_rmse_progress.png)

- failure group delta progress  
  ![failure group delta progress](/C:/Users/서동준/IdeaProjects/FAF5.7.1_2018-2024/outputs/scenario_model/model_progress_visuals/02_failure_group_delta_progress.png)

- failure group RMSE comparison  
  ![failure group rmse comparison](/C:/Users/서동준/IdeaProjects/FAF5.7.1_2018-2024/outputs/scenario_model/model_progress_visuals/03_failure_group_rmse_comparison.png)

해석 포인트:
- baseline 대비 final model이 validation, 2023, 2024에서 모두 개선됐는지
- 어떤 실패군은 개선됐고 어떤 실패군은 남았는지
- 이후 시나리오 해석의 기반이 되는 baseline이 왜 신뢰할 만한지

### A-2. 최종 보강 후보와 NGL 포트폴리오

이 섹션은 최종 결과를 한눈에 보여주는 핵심 자료다.

- 최종 보강 후보 tier 정적 지도  
  ![final reinforcement tier map](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/final_reinforcement_presentation_assets/01_final_reinforcement_tier_map.png)

- 최종 보강 후보 vs NGL overlay 정적 지도  
  ![final reinforcement ngl overlay](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/final_reinforcement_presentation_assets/02_final_reinforcement_ngl_overlay_map.png)

- 최종 후보군 표  
  ![final reinforcement candidate table](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/final_reinforcement_presentation_assets/03_final_reinforcement_candidate_table.png)

interactive 지도:
- [최종 보강 후보 tier 지도](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/final_reinforcement_presentation_assets/01_final_reinforcement_tier_map.html)
- [최종 보강 후보 vs NGL 지도](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/final_reinforcement_presentation_assets/04_final_reinforcement_candidate_map.html)

해석 포인트:
- 어떤 주가 신규 확장 후보인지
- 어떤 주가 내부 보강 후보인지
- 현재 NGL 포트폴리오와 어디서 겹치고 어디서 차이가 나는지

### A-3. 시나리오 family 교차 비교

이 두 그래프는 각 shock family를 따로 보지 않고, **반복적으로 살아남는 후보**가 누구인지를 보여준다.

- cross-family candidate counts  
  ![cross family candidate counts](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/cross_family_comparison/01_cross_family_candidate_counts.png)

- cross-family presence heatmap  
  ![cross family presence heatmap](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/cross_family_comparison/02_cross_family_presence_heatmap.png)

해석 포인트:
- `NC`, `VA`, `WA`, `MN`, `NE`, `KS`, `CO`가 왜 중요하게 남는지
- 특정 family에만 뜨는 후보와 모든 family에서 반복되는 후보를 어떻게 구분하는지

### A-4. 자연재해 트랙

자연재해는 broad network stress를 보여주는 family였고, 지도는 세부 hazard별로 분리해 보는 것이 중요하다.

interactive 지도:
- [자연재해 통합 선택 지도](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/natural_disaster_visuals/01_natural_disaster_combined_selection_map.html)
- [Snow / Heat 선택 지도](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/natural_disaster_visuals/02_snow_heat_selection_map.html)
- [Tornado 선택 지도](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/natural_disaster_visuals/03_tornado_selection_map.html)
- [Wildfire / Smoke 선택 지도](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/natural_disaster_visuals/04_wildfire_smoke_selection_map.html)

해석 포인트:
- 어떤 주가 자연재해에서 반복적으로 노출되는지
- 왜 `AR`이 fallback hub처럼 보였는지
- 자연재해가 왜 전체 네트워크를 가장 넓게 흔드는 family였는지

### A-5. 내수 문제 트랙

내수 문제는 건설/인프라, 병목/혼잡, broader demand를 따로 봐야 의미가 있다.

- domestic issue loss comparison  
  ![domestic issue loss comparison](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/domestic_issue_visuals/01_domestic_issue_loss_comparison.png)

- domestic issue candidate overlap  
  ![domestic issue candidate overlap](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/domestic_issue_visuals/02_domestic_issue_candidate_overlap.png)

interactive 지도:
- [Domestic issue state selection](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/domestic_issue_visuals/03_domestic_issue_state_selection_map.html)
- [Construction / Infrastructure vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/domestic_issue_overlap_maps/01_construction_ngl_overlap.html)
- [Domestic Bottleneck vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/domestic_issue_overlap_maps/02_bottleneck_ngl_overlap.html)
- [Common Domestic Candidates vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/domestic_issue_overlap_maps/03_common_domestic_ngl_overlap.html)

해석 포인트:
- 왜 construction shock와 bottleneck shock를 따로 봐야 하는지
- `NC`, `VA`가 내수 family 안에서 왜 반복 등장하는지
- broader domestic demand를 넣었을 때도 동부/동남부 보강축이 유지되는지

### A-6. Grain supply 트랙

곡물은 전체망 broad shock보다 **commodity-specific supply shock**로 보는 것이 더 중요하다.

- grain supply loss profile  
  ![grain supply loss profile](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/grain_supply_visuals/01_grain_supply_loss_profile.png)

- grain supply next15 touch  
  ![grain supply next15 touch](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/grain_supply_visuals/03_grain_supply_next15_touch.png)

interactive 지도:
- [Grain supply state selection](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/grain_supply_visuals/02_grain_supply_state_selection_map.html)
- [Grain supply vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/grain_supply_overlap_maps/01_grain_supply_ngl_overlap.html)

해석 포인트:
- 왜 grain supply는 전체 네트워크보다 cereal-specific loss가 훨씬 크게 나오는지
- 왜 `MN`, `NE`, `KS`가 surviving grain core인지
- 왜 `WA`, `OR`가 export support axis로 읽히는지

### A-7. Trade / external 트랙

trade family는 1차 export-only 버전보다, enhanced 버전 자료를 기준으로 보는 것이 중요하다.

- trade / external loss profile  
  ![trade external loss profile](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_visuals/01_trade_external_loss_profile.png)

- trade / external next15 touch  
  ![trade external next15 touch](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_visuals/03_trade_external_next15_touch.png)

interactive 지도:
- [Trade / external state selection](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/trade_external_visuals/02_trade_external_state_selection_map.html)
- [Enhanced trade vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/01_enhanced_trade_ngl_overlap.html)
- [Domestic broad demand vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/02_domestic_broad_demand_ngl_overlap.html)
- [Common enhanced candidates vs NGL](/C:/Users/서동준/Downloads/FAF5.7.1_2018-2024/outputs/scenario_model/enhanced_scenario_overlap_maps/03_common_enhanced_candidates_ngl_overlap.html)

해석 포인트:
- 왜 enhanced trade는 `imports + containers + partner-country dependence`를 반영해야 했는지
- 왜 `GA`, `NJ`가 trade gateway reinforcement로 새롭게 뜨는지
- 왜 `NC`, `VA`는 trade와 domestic을 가로질러 반복되는지

### A-8. 발표용으로 가장 우선순위가 높은 시각자료

시간이 없어서 모든 시각자료를 다 볼 수 없다면, 다음 순서로 보는 것이 좋다.

1. 모델 일반화 안정화 3종 그래프
2. 최종 보강 후보 tier 지도
3. 최종 보강 후보 vs NGL overlay 지도
4. cross-family heatmap
5. 자연재해 통합 선택 지도
6. domestic issue state selection
7. enhanced trade vs NGL

즉 이 부록은 단순히 이미지를 모아둔 것이 아니라,  
**발표 구성과 해석 순서까지 염두에 둔 시각자료 index**로 사용할 수 있다.
