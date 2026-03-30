import numpy as np
import json
import uuid
import chromadb
from chromadb.utils import embedding_functions
import openai
from collections import Counter


openai.api_key = "" 


# 初始化 ChromaDB (本地持久化存储)
chroma_client = chromadb.PersistentClient(path="./music_vectordb")

# 定义一个简单的自定义 Embedding 函数
# 在实际生产中，建议使用预训练的音乐模型（如 MusicBERT）
# 使用一个基于音程直方图的简单实现来演示逻辑。
class MusicIntervalEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input_texts: chromadb.Documents) -> chromadb.Embeddings:
        embeddings = []
        for text in input_texts:
            # 1. 解析文本为 MIDI 数字 (简化处理)
            try:
                # 假设输入是 "60, 64, 67" 这种格式
                midi_notes = [int(n.strip()) for n in text.split(',')]
            except:
                # 处理异常数据
                embeddings.append([0.0] * 12) 
                continue

            if len(midi_notes) < 2:
                embeddings.append([0.0] * 12)
                continue

            # 2. 计算音程序列 (Relative Pitch)
            intervals = np.diff(midi_notes)

            # 3. 创建音程直方图 (特征向量)
            # 我们只关心 12 个半音内的相对关系 (忽略八度偏差)
            interval_counts = Counter([abs(i) % 12 for i in intervals])
            
            # 构建一个 12 维的向量
            feature_vector = np.zeros(12)
            for interval, count in interval_counts.items():
                if interval < 12:
                    feature_vector[interval] = count
            
            # 归一化，使其成为单位向量 (利于余弦相似度计算)
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm
            
            embeddings.append(feature_vector.tolist())
        return embeddings

# 实例化 Embedding 函数
music_ef = MusicIntervalEmbeddingFunction()

# 创建或获取集合
collection = chroma_client.get_or_create_collection(
    name="song_phrases", 
    embedding_function=music_ef,
    metadata={"hnsw:space": "cosine"} # 使用余弦相似度
)


# ==========================================
# 2. 数据处理与存入阶段 (Data Ingestion)
# ==========================================

# 模拟：将音符文本转换为 MIDI 数字的辅助函数
# 实际生产中你需要一个完善的映射表 (G6->103, D5->74 等)
def note_to_midi(note_str):
    # 简化的映射
    mapping = {
        "C4": 60, "C#4": 61, "D4": 62, "D#4": 63, "E4": 64, "F4": 65, "F#4": 66, "G4": 67,
        "G6": 103, "D5": 74, "E-4": 63, "E-3": 51, "G5": 79, "G2": 43, "C2": 36
    }
    # 处理和弦如 "0.4.7" (这里简化的取第一个音，实际应更复杂)
    if '.' in note_str:
        note_str = note_str.split('.')[0]
        
    return mapping.get(note_str, 60) # 找不到默认给 C4

def process_and_store_songs(raw_songs_data):
    """
    处理解析后的歌曲数据并存入向量库。
    处理方式：将每首歌切分成固定长度的“旋律短语(Phrases)”存入。
    """
    phrase_length = 8  # 每个短语包含 8 个音符
    
    for song_name, raw_notes in raw_songs_data.items():
        print(f"正在处理歌曲: {song_name}...")
        
        # 1. 转换为 MIDI 数字序列
        midi_sequence = [note_to_midi(n) for n in raw_notes]
        
        # 2. 切分短语 (Sliding Window)
        ids = []
        documents = []
        metadatas = []
        
        for i in range(0, len(midi_sequence) - phrase_length + 1, 4): # 步长为4
            phrase = midi_sequence[i : i + phrase_length]
            
            # 将 MIDI 数字转为文本存储，便于 Embedding 函数处理
            phrase_str = ",".join(map(str, phrase))
            
            ids.append(f"{song_name}_{i}")
            documents.append(phrase_str)
            metadatas.append({
                "song_name": song_name,
                "start_index": i,
                "raw_notes_segment": ",".join(raw_notes[i : i + phrase_length])
            })
        
        # 3. 批量存入 ChromaDB
        if ids:
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            print(f"  已存入 {len(ids)} 个旋律片段。")

# ==========================================
# 3. Agent 检索与续写阶段 (Generation)
# ==========================================

def agent_composing_workflow(user_input_notes):
    """
    Agent 核心流程：输入->检索->构建Prompt->模型生成
    """
    print(f"\n--- Agent 开始工作 ---")
    print(f"用户输入旋律: {user_input_notes}")
    
    # 1. 预处理用户输入 (转为 MIDI)
    user_midi = [note_to_midi(n) for n in user_input_notes]
    user_midi_str = ",".join(map(str, user_midi))
    
    # 2. 向量数据库匹配相似度最高的前十条
    results = collection.query(
        query_texts=[user_midi_str],
        n_results=10,
        include=["metadatas", "documents", "distances"]
    )
    
    # 3. 解析检索结果，构建 Prompt 的上下文
    context_segments = []
    print("\n[检索到相似片段]:")
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        raw_segment = meta['raw_notes_segment']
        song_name = meta['song_name']
        
        print(f"  - 来自《{song_name}》, 相似度(距离): {dist:.4f}: {raw_segment}")
        context_segments.append(f"片段 {i+1} (来自《{song_name}》): {raw_segment}")
    
    context_text = "\n".join(context_segments)
    
    # 4. 构建大模型 Prompt
    prompt = f"""
你是一个精通音乐理论和作曲的 AI 助手。

我给你一段用户输入的音乐旋律开头（MIDI音符名格式），以及从数据库中检索到的 10 段在旋律走向上与其相似的经典音乐片段。

【用户输入的旋律开头】：
{",".join(user_input_notes)}

【参考的相似音乐片段】：
{context_text}

【你的任务】：
1. 分析用户输入的旋律开头的风格、节奏和调性特征。
2. 研究我提供的 10 段参考片段，找出它们在旋律发展、和声走向上的共同规律。
3. 根据这些规律，续写用户提供的旋律，创作出接下来约 16 到 24 个音符。
4. 保持风格一致，使其听起来像是一首完整的曲子。

【输出格式】：
仅输出续写的音符序列，用逗号分隔，不要包含任何解释性文字。
例如：G5, F#5, E5, D5, C5, ...
"""

    # 5. 调用大模型生成
    print("\n正在请求大模型生成续写...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", # 或者 "gpt-4"
            messages=[
                {"role": "system", "content": "你是一个专业的作曲家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # 保持一定的创造性
        )
        
        generated_notes = response.choices[0].message.content.strip()
        
        print("\n=== 大模型续写结果 ===")
        print(generated_notes)
        print("=======================")
        
    except Exception as e:
        print(f"调用大模型失败: {e}")


# ==========================================
# 4. 运行演示
# ==========================================

if __name__ == "__main__":
    # --- 模拟数据解析结果 (你提供的数据) ---
    parsed_songs = {
        "Example_Song_A": [
            "G6", "7.10", "E-4", "E-3", "2.3.7.10", "D6", "2.7", "G5", "G4", "10.2", 
            "G4", "G5", "F3", "10", "D5", "G5", "D5", "G4", "F3", "D4", "2.7", "G4"
        ],
        "Standard_Major_Scale": [
            "C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "B4", "A4", "G4", "F4"
        ]
    }
    
    # 1. 执行存入 (第一次运行后可以注释掉，数据已持久化)
    # 确保数据库是空的，或者做去重处理，这里演示直接添加
    if collection.count() == 0:
        process_and_store_songs(parsed_songs)
    
    # 2. 模拟 Agent 输入 (用户输入的旋律开头)
    # 这里我们输入一个和 Example_Song_A 开头有点像，但绝对音高不同的旋律
    # Example_Song_A 开头大概是 G6, D6, G5... (高音)
    # 我们输入 C4, G3, C3 (低音，但走势接近)
    user_input = ["G4", "E-4", "D4", "G3"] 
    
    # 3. 运行 Agent 流程
    agent_composing_workflow(user_input)