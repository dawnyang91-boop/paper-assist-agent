"""
Neo4j 最小插入测试脚本。

运行前在 .env 中配置：
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j

执行：
python scripts/test_neo4j_insert.py
"""

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import get_config


def main() -> int:
    config = get_config()
    if not config.neo4j_uri or not config.neo4j_username or not config.neo4j_password:
        print("缺少 Neo4j 配置，请在 .env 中设置 NEO4J_URI、NEO4J_USERNAME、NEO4J_PASSWORD。")
        return 1

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("未安装 neo4j driver，请先运行：pip install neo4j")
        return 1

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_username, config.neo4j_password),
    )
    database = config.neo4j_database or "neo4j"

    cypher = """
    MERGE (senet:Entity {name: "SENet"})
      ON CREATE SET senet.type = "model", senet.created_by = "test_neo4j_insert"
    MERGE (se:Entity {name: "SE block"})
      ON CREATE SET se.type = "method", se.aliases = ["Squeeze-and-Excitation"]
    MERGE (attention:Entity {name: "通道注意力"})
      ON CREATE SET attention.type = "concept"
    MERGE (recalibration:Entity {name: "特征重标定"})
      ON CREATE SET recalibration.type = "concept"
    MERGE (senet)-[:USES {source: "example"}]->(se)
    MERGE (se)-[:MODELS {source: "example"}]->(attention)
    MERGE (se)-[:ENABLES {source: "example"}]->(recalibration)
    RETURN senet.name AS model, se.name AS method, attention.name AS concept
    """

    with driver:
        records, _, _ = driver.execute_query(cypher, database_=database)
        for record in records:
            print(f"插入成功：{record['model']} -> {record['method']} -> {record['concept']}")

        verify_records, _, _ = driver.execute_query(
            """
            MATCH path = (:Entity {name: "SENet"})-[*1..2]-(:Entity {name: "通道注意力"})
            RETURN length(path) AS distance
            ORDER BY distance ASC
            LIMIT 1
            """,
            database_=database,
        )
        if verify_records:
            print(f"验证成功：SENet 到 通道注意力 的最短路径长度为 {verify_records[0]['distance']}")
            return 0

    print("插入后未查到验证路径，请检查 Neo4j 数据库。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
