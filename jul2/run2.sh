rk=$1
echo rank=$rk
for data in flickr ogbn-arxiv reddit 
#ogbn-products
do
    python -m torch.distributed.run --nnodes=2 --nproc_per_node=2 --node_rank=$rk --master_addr='10.242.2.105' --master_port='12355'  graphsaint_ddp.py  --dataset $data --log ddp_kp_$data.csv --batch_size 2000 --num_subgs 32 --load
done
