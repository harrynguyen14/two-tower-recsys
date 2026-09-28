# Ket qua analyze.py

- chay luc: 2026-09-28 15:19:31
- muc da chay: 4
- seed: 0  |  fingerprint du lieu: `d2e6024671ba`
- thoi gian: 5s
- **CO 1 DONG LECH [!] — doc ky truoc khi trich dan.**

`do=` la gia tri script vua tinh. `ky vong=` la so da bao cao trong session
09-23/24. Dong co `[!]` nghia la hai gia tri lech qua nguong — hoac du lieu da
doi, hoac so bao cao sai. Khong bo qua.

```
=== ANALYZE (seed=0, ky vong = so da bao cao 09-23/24) ===

4. NGAN vs DAI vs ORACLE (lich su>=50, 1 pos + 199 neg)
[nap] 1,436,609 interaction, 7,583 item, 0.2s

  n=15000
     ngan     HR@10=0.2627 (ky vong 0.2617)   MRR=0.2293 (ky vong 0.2301)
     dai      HR@10=0.2037 (ky vong 0.2042)   MRR=0.2002 (ky vong 0.2004)
     cong     HR@10=0.1997 (ky vong 0.2001)   MRR=0.1944 (ky vong 0.1946)
     max      HR@10=0.2543 (ky vong 0.0000)   MRR=0.2208 (ky vong 0.0000)
     nhan     HR@10=0.1993 (ky vong 0.0000)   MRR=0.1959 (ky vong 0.0000)
     pop      HR@10=0.3085 (ky vong 0.3087)   MRR=0.1422 (ky vong 0.1427)
     oracle   HR@10=0.3449 (ky vong 0.3438)   MRR=0.3184 (ky vong 0.3191)
      => CONG DEU te hon CA HAI thanh phan: tron TINH pha tin hieu
      => ORACLE hon ngan han +0.089 MRR: tran cua viec chon DONG

=== xong trong 5s. Moi dong co [!] la mot LECH can giai thich. ===
```
