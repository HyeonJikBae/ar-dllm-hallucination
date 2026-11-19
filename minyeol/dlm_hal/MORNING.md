# 야간 결과 (2026-11-17 01:53)

## 실행 기록
```
11-16 22:43:25 시작
11-16 22:43:25 e7 파일럿 시작 gpu1 (시도 1)
11-16 22:44:27 e7 파일럿 통과
11-16 22:44:48 e10 파일럿 실패 (e10_pilot.log 확인)
11-16 22:44:48 e7 본 실행 시작 gpu1
11-16 22:47:00 e7 본 실행 실패 rc143, 재시도 (e7 은 이어서 함)
11-16 22:50:25 e9 파일럿 시작 gpu1 (시도 1)
11-16 22:52:00 e7 파일럿 시작 gpu1 (시도 2)
11-16 22:52:14 e7 파일럿 통과
11-16 22:52:44 e10 파일럿 통과
11-16 22:52:44 e7 본 실행 시작 gpu1
11-16 22:53:16 e9 파일럿 통과
11-16 22:56:17 e9 본 실행 시작 gpu1
11-16 23:44:28 e8 파일럿 시작 gpu0 (시도 1)
11-16 23:46:47 e8 파일럿 통과
11-16 23:46:47 e8 본 실행 시작 gpu0
11-17 00:32:33 e7 본 실행 끝
11-17 01:06:48 e8 본 실행 끝
11-17 01:30:51 e9 본 실행 끝
11-17 01:30:53 보고 E6 E8 E9 rc1
11-17 01:53:13 E10 rc0
```

## E6, E8, E9
```
Traceback (most recent call last):
  File "/NAS/minyeol/ar-dllm-hallucination/followup_mk/report_e6_e8_e9.py", line 30, in <module>
    json.dump(out, open(f"{O}/report_e6_e8_e9.json", "w"), ensure_ascii=False, indent=1)
  File "/home/minyeol/.conda/envs/py312/lib/python3.12/json/__init__.py", line 179, in dump
    for chunk in iterable:
                 ^^^^^^^^
  File "/home/minyeol/.conda/envs/py312/lib/python3.12/json/encoder.py", line 432, in _iterencode
    yield from _iterencode_dict(o, _current_indent_level)
  File "/home/minyeol/.conda/envs/py312/lib/python3.12/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/home/minyeol/.conda/envs/py312/lib/python3.12/json/encoder.py", line 406, in _iterencode_dict
    yield from chunks
  File "/home/minyeol/.conda/envs/py312/lib/python3.12/json/encoder.py", line 439, in _iterencode
    o = _default(o)
        ^^^^^^^^^^^
  File "/home/minyeol/.conda/envs/py312/lib/python3.12/json/encoder.py", line 180, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type bool is not JSON serializable
```

## E10 감지
```
E7 행 9950, 사용 9950
{'n': 9950, 'mid': 5950, 'zero': 2000, 'five': 2000, 'mid_이번정답률': np.float64(47.6), '저장라벨일치': np.float64(93.8)}
최종 답 복원 일치 100.0
진동 {'0/5': {'n': 2000, '중간에_정답_나옴': np.float64(0.7), '나왔다가_사라짐': np.float64(0.5)}, '1~4/5 정답': {'n': 2833, '중간에_정답_나옴': np.float64(99.1), '나왔다가_사라짐': np.float64(0.0)}, '1~4/5 오답': {'n': 3117, '중간에_정답_나옴': np.float64(22.1), '나왔다가_사라짐': np.float64(22.1)}, '5/5': {'n': 2000, '중간에_정답_나옴': np.float64(99.9), '나왔다가_사라짐': np.float64(0.4)}}
  투영 step 0 끝
  투영 step 4 끝
  투영 step 8 끝
T1 사실 {'상위': [('가운데K L27 prompt_end s0', 0.767), ('가운데K L27 prompt_end s4', 0.728), ('가운데K L27 prompt_end s8', 0.714), ('가운데K L12 q_last s8', 0.603), ('가운데K L8 q_last s8', 0.589), ('가운데K L8 q_last s0', 0.588), ('가운데K L12 q_last s4', 0.584), ('가운데K L4 q_last s4', 0.58)], '겹밖최선': {'평균': 0.767, '고른점수': ['가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0']}}
T2 질문 {'상위': [('가운데K L27 prompt_end s0', 0.743), ('조합[가운데K x 첫4칸logprob(-)] L27 prompt_end s0', 0.733), ('조합[가운데K x 첫4칸logprob(-)] L27 prompt_end s4', 0.72), ('조합[가운데K x 첫4칸logprob(-)] L27 prompt_end s8', 0.714), ('가운데K L27 prompt_end s4', 0.709), ('가운데K L27 prompt_end s8', 0.695), ('조합[가운데K x Bw(-)] L27 prompt_end s0', 0.692), ('조합[가운데K x Bw(-)] L27 prompt_end s4', 0.624)], '겹밖최선': {'평균': 0.743, '고른점수': ['가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0', '가운데K L27 prompt_end s0']}}
T3 1~4/5 안 {'상위': [('Bw L24 prompt_end s8', 0.702), ('Bw L20 prompt_end s8', 0.699), ('중간답 종류(-) s8', 0.68), ('Bw L20 prompt_end s4', 0.678), ('Bw L8 prompt_end s8', 0.678), ('Bw L24 prompt_end s4', 0.675), ('K L8 prompt_end s8', 0.675), ('K L12 prompt_end s8', 0.672)], '겹밖최선': {'평균': 0.704, '고른점수': ['Bw L24 prompt_end s8', 'Bw L24 prompt_end s8', 'Bw L24 prompt_end s8', 'Bw L24 prompt_end s8', 'Bw L24 prompt_end s8']}}
probe 0 12 {'T1': 0.732, 'T2': 0.709, 'T3': 0.633, 'T3_같은사실': 0.626}
probe 0 20 {'T1': 0.788, 'T2': 0.755, 'T3': 0.67, 'T3_같은사실': 0.665}
probe 0 27 {'T1': 0.731, 'T2': 0.712, 'T3': 0.643, 'T3_같은사실': 0.667}
probe 8 12 {'T1': 0.757, 'T2': 0.745, 'T3': 0.764, 'T3_같은사실': 0.733}
probe 8 20 {'T1': 0.789, 'T2': 0.774, 'T3': 0.782, 'T3_같은사실': 0.744}
probe 8 27 {'T1': 0.755, 'T2': 0.742, 'T3': 0.751, 'T3_같은사실': 0.734}
게이팅 {'1~4/5 가중 정답률': {'base': 48.7, '모두_s8_31': 46.2, '모두_B_c1': 47.4, '모두_B_c0_5': 51.2, '감지게이트': 48.2, '오라클_s8_31': 52.6}, '전체 가중 정답률': {'base': 14.52, '감지게이트': 14.47, '모두_s8_31': 14.79}, '고른점수_q': [('가운데K L27 prompt_end s0', np.float64(0.02)), ('가운데K L27 prompt_end s0', np.float64(0.02)), ('가운데K L27 prompt_end s0', np.float64(0.0)), ('가운데K L27 prompt_end s0', np.float64(0.08)), ('가운데K L27 prompt_end s0', np.float64(0.02))], '민 비율': np.float64(3.0)}
판정 {'H10_1': True, 'H10_2': False, 'H10_3': False, 'H_G': False}
DONE
```
