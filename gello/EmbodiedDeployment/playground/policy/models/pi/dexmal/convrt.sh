# 将openpi的pi0权重转换为dexmal权重

# # 从 JAX 格式转换
# echo "Converting from JAX weights..."
# python3 playground/policy/models/pi/dexmal/convert_from_jax.py \
#    --jax_path /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero\
#    --output /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero/dexmal/checkpoint_jax.pkl\
#    --prompt "pick apple from the basket"\
#    --tokenizer_path /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/paligemma-3b-pt-224

# # # 从 PyTorch 格式转换
# echo "Converting from PyTorch weights..."
# python3 playground/policy/models/pi/dexmal/convert_from_torch.py \
#    --pytorch_path /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero/torch/model.safetensors\
#    --output /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero/dexmal/checkpoint_torch.pkl\
#    --prompt "pick apple from the basket"\
#    --tokenizer_path /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/paligemma-3b-pt-224

# # 对比两个转换结果是否对齐
echo "Comparing converted weights..."
python3 playground/policy/models/pi/dexmal/compare_weights.py \
   --jax_weights /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero/dexmal/checkpoint_jax.pkl\
   --torch_weights /mnt/phwfile/efm_t/zhuyangkun_tmp_need_del/model/publish/pi0_base_libero/dexmal/checkpoint_torch.pkl\
   --output compare_weights.md

