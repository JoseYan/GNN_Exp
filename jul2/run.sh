for data in flickr ogbn-arxiv reddit #ogbn-products
do
    python graphsaint_sequential.py --dataset $data --batch_size 2000   --num_subgs 32 --log ddp_kp_$data.csv --load
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_addr='127.0.0.1' --master_port='12355'  graphsaint_ddp.py  --dataset $data --log ddp_kp_$data.csv --batch_size 2000 --num_subgs 32 --load
done
