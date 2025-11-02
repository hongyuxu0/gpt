from locust import HttpUser, task, between
import json
import random

# 测试用例：随机选择提示词
PROMPTS = [
    "解释什么是人工智能",
    "写一首关于春天的诗",
    "如何学习Python编程",
    "简述相对论的核心思想",
    "推荐一部好看的科幻电影并说明理由",
    "什么是区块链技术",
    "如何提高睡眠质量",
    "解释量子计算的基本原理"
]


class GPTInferenceUser(HttpUser):
    """Locust压测用户类"""
    wait_time = between(1, 3)  # 每个用户请求间隔1-3秒

    @task(1)
    def test_generate(self):
        """测试生成接口"""
        # 随机选择提示词和参数
        prompt = random.choice(PROMPTS)
        max_gen_len = random.randint(50, 150)
        top_k = random.randint(30, 100)
        temperature = round(random.uniform(0.5, 1.0), 1)

        # 发送请求
        self.client.post(
            "/generate",
            data=json.dumps({
                "prefix": prompt,
                "max_gen_len": max_gen_len,
                "top_k": top_k,
                "temperature": temperature
            }),
            headers={"Content-Type": "application/json"}
        )

    @task(0.1)
    def test_health(self):
        """测试健康检查接口（低优先级）"""
        self.client.get("/health")

# 压测说明：
# 执行命令：locust -f load_test.py -u 50 -r 10 -t 10m --host http://localhost:8000
# 参数说明：
# -u 50：并发用户数50
# -r 10：每秒新增10个用户
# -t 10m：压测持续10分钟
