# 二分探索で適切なバッチサイズを探す

import subprocess

def C(batch_size,baseline):
    com = f"python train_mae.py --input_size=256 --epoch=1 --batch_size={batch_size} --baseline={baseline} --batch_size_search"
    args = com.split(" ")
    try:
        _ = subprocess.check_call(args,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        return True
    except:
        return False

baselines=["linear","attn","lambda"]

def binsearch(baseline):
    l, r = 1, 1<<10
    while r - l > 1:
        mid = (l+r) // 2
        print(f"batch_size={mid} => ",end='',flush=True)

        ok = C(mid,baseline)
        print("OK" if ok else "NG", flush=True)
        
        if ok: l = mid
        else: r = mid

    return l


for bl in baselines:
    print(f"====== baseline : {bl} ======")
    print(f"\n=> batch_size = {binsearch(bl)}\n")