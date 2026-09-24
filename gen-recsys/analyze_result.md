# Ket qua analyze.py

- chay luc: 2026-09-24 09:11:01
- muc da chay: 1, 2, 3, 4, 5, 6, 7, 8
- seed: 0  |  fingerprint du lieu: `d2e6024671ba`
- thoi gian: 8s
- **CO 1 DONG LECH [!] — doc ky truoc khi trich dan.**

`do=` la gia tri script vua tinh. `ky vong=` la so da bao cao trong session
09-23/24. Dong co `[!]` nghia la hai gia tri lech qua nguong — hoac du lieu da
doi, hoac so bao cao sai. Khong bo qua.

```
=== ANALYZE (seed=0, ky vong = so da bao cao 09-23/24) ===

1. PARSE TAG — bug multi-label da sua
[nap] 1,436,609 interaction, 7,583 item, 0.0s

    item co 1 tag                              do=5696  ky vong=5696
    item co 2 tag                              do=1759  ky vong=1759
    item co 3 tag                              do=32  ky vong=32
    item khong tag                             do=96  ky vong=96
    tag duy nhat                               do=46  ky vong=46
    nhan cu (111 = HONG)                       do=111  ky vong=111

2. VONG DOI ITEM (mo ta, toan log)
    tuoi doi p50 (ngay) — VO DUNG              do=26.62  ky vong=26.62
    age_at_event p25 (ngay)                    do=0.68  ky vong=0.68
    age_at_event p50 (ngay)                    do=2.26  ky vong=2.26
    age_at_event p90 (ngay)                    do=19.90  ky vong=19.90
    corr(age, log m_j) — PHAI ~0               do=-0.0323  ky vong=-0.0320
      => age KHONG du thua voi popularity: dieu kien chan PASS

3. VI TRI TIN HIEU (46 tag dung) — bang chung TRUNG TAM
  nguong            n   CHI dai  CHI ngan   ca hai    khong
  >=30         20000    0.1990    0.0954   0.4921   0.2135   (ky vong CHI dai=0.1990)
  >=50         20000    0.2631    0.0466   0.5381   0.1523   (ky vong CHI dai=0.2631)
  >=100        20000    0.3358    0.0181   0.5563   0.0898   (ky vong CHI dai=0.3358)
  >=200         6700    0.3790    0.0072   0.5628   0.0510   (ky vong CHI dai=0.3790)
      => moi nguong: ngan han mu nhieu gap 2-53 lan dai han mu

4. NGAN vs DAI vs ORACLE (lich su>=50, 1 pos + 199 neg)
  n=15000
     ngan     HR@10=0.2627 (ky vong 0.2617)   MRR=0.2293 (ky vong 0.2301)
     dai      HR@10=0.2037 (ky vong 0.2042)   MRR=0.2002 (ky vong 0.2004)
     cong     HR@10=0.1997 (ky vong 0.2001)   MRR=0.1944 (ky vong 0.1946)
     pop      HR@10=0.3085 (ky vong 0.3087)   MRR=0.1422 (ky vong 0.1427)
     oracle   HR@10=0.3449 (ky vong 0.3438)   MRR=0.3184 (ky vong 0.3191)
      => CONG DEU te hon CA HAI thanh phan: tron TINH pha tin hieu
      => ORACLE hon ngan han +0.089 MRR: tran cua viec chon DONG

5. BURST + THAT BAI CUA GATE THU CONG
    entropy 10 item gan nhat (nats)            do=1.896  ky vong=1.895
    entropy item 11-200 (nats)                 do=2.894  ky vong=2.894
    JS-divergence(ngan||dai)                   do=0.328  ky vong=0.326
      => ngan han hep hon 34.5% — CO burst that
      NHUNG: entropy chia quartile cho ti le ngan-thang .618/.643/.664/.654
      => gan PHANG, con hoi NGUOC. Gate thu cong THAT BAI -> phai de mo hinh HOC

6. PMI (train-only, khong leak)
    PMI median                                 do=+1.404  ky vong=+1.404
    ti le PMI > 0                              do=0.8862  ky vong=0.8860
    corr(PMI, log m_j)                         do=-0.4984  ky vong=-0.4980
    R2(PMI, chung tag) — PHAI ~1%              do=0.0104  ky vong=0.0104
      bang dense fp16: 115.0 MB (fp32 = 230.0 MB)
  n=3000
    HR@10 pop                                  do=0.3407  ky vong=0.3342
    HR@10 mix                                  do=0.5457  ky vong=0.5566
    HR@10 pmi                                  do=0.2790  ky vong=0.2838
      => PMI MOT MINH thua pop (.284<.334) — do rieng se ket luan SAI
      => pop+PMI = .557, +66%: hai tin hieu truc giao, PHAI cong THEM

7. MATTHEW EFFECT — do de BAC BO, khong phai de xac nhan
    top 10% item chiem                         do=0.5733  ky vong=0.5730
    Gini                                       do=0.7117  ky vong=0.7117
      nhom theo pop nua dau -> he so thay doi thi phan o nua sau:
      duoi p25                                 do=x1.921  ky vong=x1.921
      p25-p50                                  do=x1.329  ky vong=x1.329
      p50-p75                                  do=x1.187  ky vong=x1.187
      tren p75                                 do=x0.925  ky vong=x0.925
      => ngheo x1.92, giau x0.93: HOI QUY VE TRUNG BINH, NGUOC Matthew
      => Pure co can thiep ngau nhien nen vong phan hoi da bi cat

8. TAG vs VONG DOI
    eta2(life | tag) — PHAI thap               do=0.0425  ky vong=0.0425
    tag co >=5000 event                        do=35  ky vong=35
    toc do phai NHANH nhat                     do=3.05  ky vong=3.05
    toc do phai CHAM nhat                      do=0.27  ky vong=0.27
    chenh lech (lan)                           do=11.1  ky vong=11.1
      => tag khong quyet dinh item song bao lau (eta2 ~4%)
      => nhung quyet dinh item PHAI NHANH hay CHAM: chenh 11x
      => age phai NHAN vao content (FiLM), khong CONG

=== xong trong 8s. Moi dong co [!] la mot LECH can giai thich. ===
```
