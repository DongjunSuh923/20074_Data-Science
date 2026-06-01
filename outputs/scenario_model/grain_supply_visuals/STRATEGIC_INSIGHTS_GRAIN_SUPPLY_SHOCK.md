# STRATEGIC_INSIGHTS_GRAIN_SUPPLY_SHOCK

이 문서는 grain supply shock 시뮬레이션의 중간 해석 메모다. 풍작/흉작과 공급량 변화가 물류에 미치는 영향을 보기 위한 것으로, 자연재해가 아니라 **commodity-specific supply shock** 관점에서 정리한다.

## 핵심 결과

- cumulative network loss: 0.19%
- peak network loss: 0.75% in 2026
- cereal-only cumulative loss: 2.42%
- cereal-only peak loss: 9.91%

해석상 이 shock는 전체 네트워크보다 **곡물 계열 물동량 자체를 강하게 흔드는 narrow commodity shock**에 가깝다.

## 왜 이 주들이 선정됐나

- `IA` severe: grain production `16,179,880,000` bu, harvested area `134,060,000`, drought events `244.25`
- `TX` medium: grain production `1,145,588,000` bu, harvested area `18,484,000`, drought events `1031.75`
- `NE` medium: grain production `10,947,360,000` bu, harvested area `92,300,000`, drought events `338.0`
- `IL` mild: grain production `15,333,050,000` bu, harvested area `130,500,000`, drought events `91.0`
- `MN` mild: grain production `8,777,710,000` bu, harvested area `93,910,000`, drought events `112.5`

즉 선정 기준은 생산 기반 규모와 최근 기후 스트레스가 결합된 곳이다. 생산량이 크고, 수확면적이 넓고, 최근 drought exposure가 높은 주일수록 공급 shock에 취약한 것으로 보았다.

## 시나리오 이후 next 15의 의미

- must-have 제외 next 15: MN, NE, WA, KS, WI, CO, SD, LA, AL, OR, IA, VA, ND, SC, NC

여기서 `MN`, `NE`, `WA`, `KS`, `WI`, `CO`, `SD`가 상위권에 남는 것은, direct grain core가 흔들릴 때 **Midwest support + western redistribution** 조합이 중요하다는 뜻에 가깝다.

`IA`, `IL`, `NE`, `MN` 같은 직접 생산 핵심 주는 shock 대상과 support 후보가 부분적으로 겹치지만, `IA`와 `IL`은 직접 타격을 받아 오히려 순위에서 밀리고, 주변 support state가 상대적으로 부상하는 구조가 보인다.

## NGL 포트폴리오와의 관계

이 트랙은 NGL의 기존 footprint와 상당 부분 겹치지만, 완전히 같지는 않다. 따라서 grain shock는 trade shock처럼 coastal exposure 문제도 아니고, domestic shock처럼 corridor reliability 문제도 아니다. **농업 공급 축이 흔들릴 때 어떤 주가 cereal-support state로 살아남는가**에 더 가깝다.

## 현재 단계의 중간 판단

- grain supply shock는 total network loss는 작지만 cereal-only impact는 크다.
- 따라서 이 트랙은 전사적 네트워크 우선순위보다, commodity-specific portfolio resilience를 보는 용도로 더 적합하다.
- 향후 NGL 제안에선 `MN`, `NE`, `KS`, `WA`를 grain-support candidates로 다시 볼 필요가 있다.