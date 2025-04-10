# Docker Swarm

## 1. Rent, create VPN, and install Docker

- Rent a VPS or EC2 instance (Ubuntu 20.04) for the manager node. Config security group (if using AWS EC2) to allow 2377
- Rent a VPS or EC2 instance (Ubuntu 20.04) for the worker node
- Install Docker on both nodes

## 2. Build and push Docker image

Login to Docker Hub

```bash
docker login
```

Build and push the backend image

```bash
cd backend
docker buildx build --platform linux/amd64,linux/arm64 -t <your_dockerhub_username>/demo-swarm-backend:latest --push .

# e.g.
# docker buildx build --platform linux/amd64,linux/arm64 -t thuannan/demo-swarm-backend:v2.0.0 --push .
```

Build and push the frontend image

```bash
cd frontend
docker buildx build --platform linux/amd64,linux/arm64 -t <your_dockerhub_username>/demo-swarm-frontend:latest --push .

# e.g.
# docker buildx build --platform linux/amd64,linux/arm64 -t thuannan/demo-swarm-frontend:latest --push .
```

## 3. Set up Docker Swarm

Manager Node (manager vps/ec2)

```bash
docker swarm init
docker node ls 
```

Copy the join command and run it on worker nodes like:

```bash
docker swarm join --token <TOKEN> <IP>:<PORT>
```

Copy docker command from the manager node

```bash
scp <source file> <user>@<destination>:/home/ubuntu/
```

Pull images

```bash
docker pull <your_dockerhub_username>/demo-swarm-backend:latest
docker pull <your_dockerhub_username>/demo-swarm-frontend:latest

# e.g.
# docker pull thuannan/demo-swarm-backend:latest
# docker pull thuannan/demo-swarm-frontend:latest
```

## 4. Deploy stack

```bash
docker stack deploy -c compose.swarm.yml demo-swarm
```

## 5. Check the status

```bash
docker service ls
```

## 6. Scale the service

```bash
docker service scale <service_name>=<number_of_replicas>

# e.g.
# docker service scale demo-swarm_yolov8-api=3
# docker service scale demo-swarm_yolov8-frontend=3
```

## 7. Remove the stack

```bash
docker stack rm demo-swarm
```
