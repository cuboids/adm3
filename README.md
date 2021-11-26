# adm2
Advances in Data Mining 2

We will use the [CSR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html) matrix. Why? 'Cause the [pro implementation](https://github.com/brandonrobertz/SparseLSH/blob/main/sparselsh/lsh.py) of LSH uses it.

The [plot below](https://www.wolframalpha.com/input/?i=1205+*+%281+-+%281-.5%5Ex%29%5E%28256%2Fx%29%29+for+2+%3C+x+%3C+16) plots how many user pairs we are expected to find if we set N_SIGNATURES = 256.

![image](https://user-images.githubusercontent.com/44651818/143563233-69c841f5-f1b0-4f51-8d19-e405df506f28.png)
