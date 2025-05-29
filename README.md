# NeurSTT

**[Pattern Recognition 2025]** This is the official implementation of "Neural Spatial-Temporal Tensor Representation for Infrared Small Target Detection".

## Notification
🎉🎉🎉 May 2025: We are glad to inform that our NeurSTT is accepted by Pattern Recognition.

⭐⭐⭐ March 2025: We were invited by Dr. [Yisi Luo](https://yisiluo.github.io/) (advisor: [Prof. Deyu Meng](https://gr.xjtu.edu.cn/en/web/dymeng/1)) from XJTU to give a talk on our NeurSTT.

🔥🔥🔥 March 2025: We released the code of NeurSTT (ver 1.0).

🏠🏠🏠 August 2024: We submitted our manuscript.

## 1. Requirements

- **Python 3.8**
- **Windows 10, Ubuntu 18.04 or higher**
- **NVIDIA GeForce RTX 3090**
- **PyTorch 1.8.0 or higher**
- **More details available in requirements.txt**

## 2. Datasets

We used sequences from: [[1]](http://www.csdata.org/en/p/387/),[[2]](https://ieeexplore.ieee.org/document/10011452). Please download these datasets or set your own datasets to the `./data/` folder.

### Example Dataset Structure
```
├──./data/
│    ├── exp1
│    │    ├── images
│    │    │    ├── 000.bmp
│    │    │    ├── 001.bmp
│    │    │    ├── ...
│    ├── exp1.gt
│    │    │    ├── 000.bmp
│    │    │    ├── 001.bmp
│    │    │    ├── ...
│    ├── ...
```

Evaluation of our model on other datasets is welcome!

## 3. Commands for Using Our Code

- Install the environment according to `requirements.txt`.

- Enter the repository and run `main.py` to perform network training:
  - To run the code with default parameter settings:
    ```bash
    $ python main.py
    ```
  - To change parameters for further research:
    ```bash
    $ python main.py --model_name NeurSTT --dataset exp1 --frame 80 --gamma 0.25 --phi 5e-5 --kappa 1 --max_iter 1500
    ```

## 4. Commands for Evaluation

The results in our project have the following structure:

```
├──./result/
│    ├── exp1
│    │    ├── NeurSTT
│    │    │    ├── Seg
│    │    │    │    ├── 000.png
│    │    │    │    ├── 001.png
│    │    │    │    ├── ...
│    │    │    ├── T
│    │    │    │    ├── 000.png
│    │    │    │    ├── 001.png
│    │    │    │    ├── ...
│    │    │    ├── log.txt
│    │    ├── ...
│    ├── ...
```

- Enter the repository and run `eval.py` to perform evaluation:
```bash
$ python eval.py
```
- The evaluation log result will be saved in `eval_log.txt`:
```
├──./result/
│    ├── exp1
│    │    ├── eval_log.txt
│    ├── ...
```
## Contact
For any questions regarding this paper or the code, please feel free to reach out to [wufengyi98@163.com](wufengyi98@163.com).

## Reference
[1] Y. Luo, et al. "Neurtv: Total variation on the neural domain," arXiv preprint arXiv:2405.17241, 2024.

[2] Y. Luo, et al. "Low-Rank Tensor Function Representation for Multi-Dimensional Data Recovery," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 5, pp. 3351-3369, May 2024, doi: 10.1109/TPAMI.2023.3341688.

[3] Z. Zhang, et al. "Infrared Small Target Detection Combining Deep Spatial–Temporal Prior With Traditional Priors," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-18, 2023, Art no. 5004718, doi: 10.1109/TGRS.2023.3323339.
