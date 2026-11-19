"""E56 / E58a 보고서 (txt) 를 요약 json 에서 만든다.  repo 루트에서:  python3 followup_mk/e56_report.py"""
import json
O = "followup_mk"; D = f"{O}/e56_data"
P = json.load(open(f"{D}/prep_counts.json")); Q = json.load(open(f"{D}/para_counts.json")); S = json.load(open(f"{O}/e56_summary.json"))
pct = lambda x: "NA" if x is None else f"{100 * x:.1f}"
cis = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
M = S["main"]; C = S["contain_rule"]; H1 = M["H56.1_data"]; H2 = M["H56.2_data"]; ST = M.get("stratified_qwen_lp_q5_x_len", {})
L = []
w = L.append
w("E56 두 번째 자료: TriviaQA 다섯 문구 (Dream-v0-Base-7B 대 Qwen2.5-7B)\n")
w("1. 자료 거르기")
w(f"  TriviaQA rc.nocontext validation {P['rows_parquet']} 행 -> 질문 id 중복 제거 {P['unique_question_id']} -> WikipediaEntity {P['type_WikipediaEntity']} -> 답 5 단어 이하 {P['value_le5_words']} -> 별칭 있음 {P['has_alias']}"
  f" -> 답이 질문에 안 나옴 {P['value_not_in_question']} -> 질문 4~60 단어 {P['question_4_60_words']} -> seed 56 표본 {P['sampled']}.")
c = Q["counts"]
w(f"  Qwen3-14B (생각 끔) 다른 문구 4 개: 생성 성공 {c['gen_ok']}/{c['questions']}.  문구 검사 통과 {c['paraphrase_pass']}/{c['paraphrases']} ({pct(Q['paraphrase_pass_rate'])}%);"
  f" 실패 이유 {Q['fail_reasons']}.  네 문구 모두 통과한 사실 {c['facts_4of4_pass']} (x5 = {5 * c['facts_4of4_pass']} 행).")
w("  주의: 같은 모델의 'leak' 판정은 한 번도 '예' 가 나오지 않음 -> 답 노출은 사실상 규칙 검사 (별칭 문자열) 로만 걸렀다.  표본 30 개는 e56_data/spotcheck.txt.")
w(f"\n2. 정확도 (행 {S['n_rows']}, 사실 {S['n_facts']})")
a = M["acc"]
w(f"  Dream base {pct(a['dream_base'])}%, Dream 칸0먼저 {pct(a['dream_kan0'])}%, Qwen2.5-7B {pct(a['qwen'])}%.  문구별 Dream {a['dream_base_by_ver']}, Qwen {a['qwen_by_ver']}.")
w(f"  k/5 사실 수: Dream {M['k5']['dream']}, Qwen {M['k5']['qwen']}.  Dream 1~4/5 사실 {M['n_mid_facts']}.")
w(f"  같은 사실 안 문구 정오 일치 (Dream 1~4/5 이고 Qwen 도 섞인 사실 {M['concordance_mixed_facts']['n_fact']}): P(Dream 맞힘|Qwen 맞힘) − P(..|Qwen 틀림) = {M['concordance_mixed_facts']['mean']}")
r = M["routes"]
w(f"\n3. 경로 (Dream 1~4/5 base 오답 entity/broken {r['n_wrong']} 행): DLM 고유 (같은 문구 Qwen 맞힘) {r['n_dlm']}, 공유 {r['n_shared']} (abstain 제외 {r['n_abstain_excluded']}).  유형 {r['types']}")
w("\n4. H56.1  step 0 칸 0 금 첫 토큰 1 위 비율 (DLM 고유 > 공유, 사실 군집 CI 하한 > 0)")
g = H1["gold_top1"]
w(f"  대표 답 첫 토큰: DLM 고유 {pct(g['A']['rate'])}% 대 공유 {pct(g['B']['rate'])}%, 차이 {pct(g['diff'])} {cis(g['diff_ci'])}  -> H56.1 {'참' if M['H56.1']['verdict'] else '거짓'}.")
wf = g["within_fact"]
if wf: w(f"  같은 사실 안 (두 집단 다 있는 사실 {wf['n_fact']}): {pct(wf['diff'])} {cis(wf['ci'])}")
g2 = H1["any_alias_top1"]; w(f"  어느 별칭이든 첫 토큰: {pct(g2['A']['rate'])} 대 {pct(g2['B']['rate'])}, 차이 {pct(g2['diff'])} {cis(g2['diff_ci'])}")
w(f"  금 순위 중앙: {H1['median_rank']};  top-5 {pct(H1['gold_top5']['A']['rate'])} 대 {pct(H1['gold_top5']['B']['rate'])};  '1 위였다가 칸 0 에서 잃음' {pct(H1['lost_top1_then_slot0_not_gold']['A']['rate'])} 대 {pct(H1['lost_top1_then_slot0_not_gold']['B']['rate'])}")
w(f"  참고 1 위 비율: 1~4/5 정답 {pct(H1['top1_reference']['mid_correct'])}, 0/5 {pct(H1['top1_reference']['zero'])}, 5/5 {pct(H1['top1_reference']['five'])}.  원문 문구만: 차이 {pct(H1['original_question_only']['diff'])} {cis(H1['original_question_only']['diff_ci'])}")
w("\n5. H56.2  칸0먼저 (step 0 칸 0 argmax 를 먼저 확정) 의 오답 복구율 (DLM 고유 > 공유, 하한 > 0)")
w(f"  DLM 고유 {pct(H2['A']['rate'])}% (n {H2['A']['n']}) 대 공유 {pct(H2['B']['rate'])}% (n {H2['B']['n']}), 차이 {pct(H2['diff'])} {cis(H2['diff_ci'])}  -> H56.2 {'참' if M['H56.2']['verdict'] else '거짓'}.")
if H2["within_fact"]: w(f"  같은 사실 안: {pct(H2['within_fact']['diff'])} {cis(H2['within_fact']['ci'])} (사실 {H2['within_fact']['n_fact']})")
w(f"  유형별: entity 차이 {pct(H2['by_type']['entity']['diff'])} {cis(H2['by_type']['entity']['diff_ci'])}, broken {pct(H2['by_type']['broken']['diff'])} {cis(H2['by_type']['broken']['diff_ci'])}")
w(f"  칸0먼저 전체 효과: {json.dumps(M['kan0_effect'], ensure_ascii=False)}")
b = M["break_mid_correct_by_qwen"]; w(f"  1~4/5 정답 행이 칸0먼저로 깨지는 비율: Qwen 틀린 문구 {pct(b['A']['rate'])} 대 Qwen 맞힌 문구 {pct(b['B']['rate'])}, 차이 {pct(b['diff'])} {cis(b['diff_ci'])}")
if ST:
    w("\n6. 층화 (E57 식: 경로 행 안에서 Qwen2.5 금 첫 토큰 log 확률 5 분위 x 금 답 토큰 수 1/2/3/4+, Mantel-Haenszel 차이, 사실 군집 CI)")
    for k_, v in ST.items():
        if isinstance(v, dict) and "mh_diff" in v: w(f"  {k_}: {pct(v['mh_diff'])} {cis(v['ci'])} (덮는 행 {pct(v['coverage'])}%)")
    w(f"  Qwen 금 log 확률 중앙 DLM 고유 / 공유: {ST.get('qwen_lp_median')}")
w("\n7. 채점 민감도 (포함 규칙: 별칭이 답 안에 단어 단위로 있으면 정답)")
w(f"  Dream 정확도 {pct(C['acc']['dream_base'])}, Qwen {pct(C['acc']['qwen'])}.  H56.1 차이 {pct(C['H56.1']['diff'])} {cis(C['H56.1']['ci'])}, H56.2 차이 {pct(C['H56.2']['diff'])} {cis(C['H56.2']['ci'])}.  주 규칙과 일치 {S['grading_agreement']}")
w(f"\n8. sanity: {S['sanity']}.  파일럿: 직접 짠 base 생성이 dloop.generate 와 답·캔버스 32/32 일치.")
w("\n9. 결과 보기 전 설계 변경 / 비고")
w("  - 채점: PopQA is_correct 규칙 그대로 + 앞 관사 지운 형태 추가 (TriviaQA 공식 정규화가 관사를 지움).  질문에 이미 나오는 별칭은 뺌.  괄호 지운 별칭은 안 넣음 (위키 넘겨주기 잡음).")
w("  - 분류: Wikidata 이름 규칙 (8 번) 은 TriviaQA 에 관계·캐시가 없어 끔 (entity 로 둠).")
w("  - '어느 별칭이든' 첫 토큰은 value 와 내용어를 공유하고 'List of ...' 가 아니며 value 보다 2 단어 이상 길지 않은 별칭만 (앞 관사·호칭 지움).")
w("  - 칸0먼저와 base 는 모든 행에서 같은 묶음 (16 행, 같은 길이) 으로 돌림.  k/5 와 경로도 이 base 로.  Qwen 은 패딩 없는 같은 길이 64 행 묶음.")
w("  - 문구 생성기에는 답을 보여주지 않음 (PopQA 와 같음); 검사기에는 보여줌.  표집 temperature 0.7, top_p 0.8, top_k 20.")
w("\n파일: followup_mk/e56_*.py, e56_data/, results_e56_dream_s*.jsonl, results_e56_qwen_s0.jsonl, e56_summary.json, logs/e56_*.log")
open(f"{O}/report_e56.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")

try: E = json.load(open(f"{O}/e58a_summary.json"))
except FileNotFoundError: E = None
if E:
    L = []; w = L.append
    w("E58 (a) 주 방법 확인: TriviaQA 다섯 문구 위 Dream (3 진 분류 -> 오답 고르기 -> 층 24 B c0.5 로 다시 생성)\n")
    w(f"행 {E['n_rows']}, 사실 {E['n_facts']}, k/5 {E['k5']}.  base 정확도 {pct(E['base_acc'])}%, 모든 행 B24 재생성 {pct(E['B24_all_acc'])}%.")
    w(f"3 진 P(1~4/5) AUROC (1~4/5 대 나머지) {E['auroc_mid_vs_rest']['3class_P']} (S1 단독 {E['auroc_mid_vs_rest']['S1']});  SC 의 1~4/5 오답 대 정답 AUROC {E['auroc_SC_mid_wrong_vs_mid_correct']}.")
    for nm, lab in (("three_class", "3 진 (주 방법)"), ("sequential_AND", "연속 2 진 AND (서술)"), ("all_B24", "참고: 모든 행 재생성")):
        x = E[nm]
        w(f"\n[{lab}] 라우팅 {x['routed']} 행.  1~4/5: 회복 {x['recovered_mid']} ({pct(x['recover_rate_mid_wrong'])}% of 오답) / 망가짐 {x['broken_mid']} ({pct(x['break_rate_mid_correct'])}% of 정답) / 순 {x['mid_net']:+d},"
          f" 변화 {x['mid_change_pts'][0]:+.2f} 점 [{x['mid_change_pts'][1]:+.2f}, {x['mid_change_pts'][2]:+.2f}]")
        w(f"   5/5: base {x['five_acc_base']}%, 변화 {x['five_change_pts'][0]:+.2f} 점 [{x['five_change_pts'][1]:+.2f}, {x['five_change_pts'][2]:+.2f}] (망가짐 {x['five_broken']} 행).  0/5 재생성 비율 {pct(x['zero_routed_rate'])}%, 0/5 이득 {x['zero_gain_rows']:+d} 행.")
        w(f"   전체 (모집단) 순 {x['pop_net_rows']:+d} 행, {x['pop_change_pts'][0]:+.2f} 점 [{x['pop_change_pts'][1]:+.2f}, {x['pop_change_pts'][2]:+.2f}].  라우팅 몫 {x['routed_share']}.  경로별 회복 {x['recover_by_route']}")
        if "thresholds_by_fold" in x: w(f"   겹별 고른 격자 번호 (None = 안 함): {x['thresholds_by_fold']}")
    w(f"\nH58.1 (1~4/5 순 변화 > 0, 하한 > 0): {'참' if E['H58.1']['verdict'] else '거짓'} {E['H58.1']}")
    w(f"H58.2 (5/5 손실 <= 1 점): {'참' if E['H58.2']['verdict'] else '거짓'} {E['H58.2']}")
    w(f"H58.3 (모집단 순 이득 > 0, 하한 > 0): {'참' if E['H58.3']['verdict'] else '거짓'} {E['H58.3']}")
    w("\ne30c_pipeline.py (PopQA) 와 맞춘 것: 특징 4 개 (칸 0 top-1 logprob, 첫 4 칸 평균, 칸 0 margin, 겹밖 S1), 학습 겹 평균·표준편차 표준화 다항 로지스틱,"
      " 전체 행 분위 10 개 격자, 겹마다 학습 겹 이득 최대 마스크 / 이득 <= 0 이면 '안 함', 연속 2 진 = LP(−칸 0 logprob) > a & S1 > b & SC > c.")
    w("남은 차이: (1) 모집단 가중 없음 (모든 행을 실제로 돌렸으므로 가중 1); (2) 겹 = 질문 id 5 겹 (seed 58); (3) K 는 관계가 없어 전체 5/5 − 0/5 (층 27 프롬프트 끝 step 0);"
      " (4) B 는 층 24 프롬프트 끝 step 0, 같은 사실 안 (맞힘 평균 − 틀림 평균, 틀림 = 정답 아님 전부) 의 사실 평균, 겹밖, 0.5 x 평균 ||h||, 모든 step;"
      " (5) 0/5 는 실제 재생성 결과; (6) SC 의 중간 답 = 각 step 의 캔버스 (가려진 칸은 그 step 예측, 확정 칸은 확정 토큰) 를 E30 과 같은 방식으로 decode.")
    w("\n파일: followup_mk/e58a.py, e56_dream.py (--steer), e56_data/folds.json, e56_data/e58_dirs.npz, results_e56_steer_s*.jsonl, e56_hid/, e58a_summary.json")
    open(f"{O}/report_e58a.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("reports written")
