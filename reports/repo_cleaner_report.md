# AI Study Repository Cleaner Report

생성일: 2026-06-09T15:23:22
분석 루트: `C:\Users\darks\Desktop\자료들\ai자료\ai`

## 1. 전체 요약
- 총 파일 수: 1273
- 총 폴더 수: 315
- 전체 용량: 408.72 MB
- Git 제외 후보: 81
- 민감정보 후보: 117
- 대용량 파일 후보: 11

주요 확장자:

- `.csv`: 277
- `.py`: 196
- `.html`: 187
- `.ipynb`: 121
- `.jsonl`: 118
- `.json`: 94
- `.png`: 40
- `.jpg`: 36
- `.md`: 32
- `.txt`: 30

## 2. 최상위 폴더별 요약

| 폴더 | 분류 | 파일 수 | 하위 폴더 수 | 용량 | 주요 확장자 | 위험 후보 | 대용량 |
|---|---|---:|---:|---:|---|---:|---:|
| `.` | 분류 불명 | 5 | 0 | 10.36 KB | [no extension]:2, .md:2, .txt:1 | 3 | 0 |
| `archive` | 문서/관리 | 1 | 1 | 4.04 KB | .md:1 | 1 | 0 |
| `assets` | 빈 폴더 | 0 | 0 | 0 B | - | 0 | 0 |
| `coursework` | Python 기초/실습 | 1204 | 266 | 405.35 MB | .csv:257, .py:195, .html:187, .jsonl:118, .ipynb:107 | 174 | 11 |
| `docs` | 문서/관리 | 3 | 0 | 6.28 KB | .md:3 | 1 | 0 |
| `experiments` | Python 기초/실습 | 9 | 6 | 65.35 KB | .ipynb:9 | 7 | 0 |
| `programming` | 웹앱/백엔드 | 19 | 17 | 16.84 KB | .md:19 | 4 | 0 |
| `projects` | 데이터 분석 | 27 | 12 | 3.17 MB | .csv:20, .ipynb:5, .md:1, .txt:1 | 4 | 0 |
| `prompt-decks` | 빈 폴더 | 0 | 0 | 0 B | - | 0 | 0 |
| `reports` | 문서/관리 | 2 | 0 | 77.83 KB | .json:1, .md:1 | 2 | 0 |
| `scripts` | 빈 폴더 | 0 | 0 | 0 B | - | 0 | 0 |
| `tools` | Python 기초/실습 | 3 | 1 | 22.93 KB | .md:2, .py:1 | 2 | 0 |

## 3. Git 제외 후보

- `coursework/AI_Study/.ipynb_checkpoints` - jupyter checkpoint
- `coursework/AI_Study/01_python/.ipynb_checkpoints` - jupyter checkpoint
- `coursework/AI_Study/01_python/ch09/__pycache__` - python cache
- `coursework/AI_Study/01_python/data/ch09_member.pkl` - model/data artifact
- `coursework/AI_Study/01_python/data/ch10.pkl` - model/data artifact
- `coursework/AI_Study/01_python/data/ch11_iris.csv.gz` - archive
- `coursework/AI_Study/01_python/data/ch11_iris.gz` - archive
- `coursework/AI_Study/01_python/data/ch15_example.db` - database
- `coursework/AI_Study/05_DeepLearning/model/02_deep.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/06binary.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/07.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/after_learning.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/before_learning.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/iris-007-loss0.7205-acc1.0000.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/noised_after_learning.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/noised_after_learning1.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model/wine.h5` - model file
- `coursework/AI_Study/05_DeepLearning/model08/mnist-019-val0.2967.h5` - model file
- `coursework/AI_Study/06_이미지처리/mnist-13-loss0.0312-val0.9922.h5` - model file
- `coursework/AI_Study/06_이미지처리/mnist-15-loss0.0311-val0.9933.h5` - model file
- `coursework/AI_Study/06_이미지처리/model/finalModel/keras_metadata.pb` - model file
- `coursework/AI_Study/06_이미지처리/model/finalModel/saved_model.pb` - model file
- `coursework/AI_Study/07_자연어처리/imdb_v2.10_skip20length80.h5` - model file
- `coursework/AI_Study/07_자연어처리/data/seq2seq.h5` - model file
- `coursework/AI_Study/08_LLM/chroma` - chroma vector database
- `coursework/AI_Study/08_LLM/chroma_upstage` - chroma vector database
- `coursework/AI_Study/08_LLM/__pycache__` - python cache
- `coursework/AI_Study/09_MachineLearning/catboost_info/learn/events.out.tfevents` - training event log
- `coursework/AI_Study/09_MachineLearning/data/ch01_arr1.pkl` - model/data artifact
- `coursework/AI_Study/09_MachineLearning/data/ch01_mlp_model.joblib` - model/data artifact
- `coursework/AI_Study/09_MachineLearning/data/ch1_mlp_model.pkl` - model/data artifact
- `coursework/AI_Study/11_flask/ch02_route_render/__pycache__` - python cache
- `coursework/AI_Study/11_flask/ch1/__pycache__` - python cache
- `coursework/AI_Study/11_flask/ch1/model/apt.joblib` - model/data artifact
- `coursework/AI_Study/11_flask/ch1/model/apt.pkl` - model/data artifact
- `coursework/AI_Study/11_flask/ch3_CRUD/venv` - virtual environment
- `coursework/AI_Study/11_flask/ch3_methods_error/__pycache__` - python cache
- `coursework/AI_Study/11_flask/ch4_jinja2/__pycache__` - python cache
- `coursework/AI_Study/12_django/ch01_hello/db.sqlite3` - sqlite database
- `coursework/AI_Study/12_django/ch01_hello/ch01/__pycache__` - python cache
- `coursework/AI_Study/12_django/ch02_wordcnt/venv` - virtual environment
- `coursework/AI_Study/12_django/ch02_wordcnt/db.sqlite3` - sqlite database
- `coursework/AI_Study/12_django/ch02_wordcnt/ch02/__pycache__` - python cache
- `coursework/AI_Study/12_django/ch02_wordcnt/home/__pycache__` - python cache
- `coursework/AI_Study/12_django/ch02_wordcnt/home/migrations/__pycache__` - python cache
- `coursework/AI_Study/12_django/ch02_wordcnt/wordcnt/__pycache__` - python cache
- `coursework/AI_Study/12_django/ch02_wordcnt/wordcnt/migrations/__pycache__` - python cache
- `coursework/AI_Study/12_django/myproject/_staticfiles` - generated static files
- `coursework/AI_Study/12_django/myproject/db.sqlite3` - sqlite database
- `coursework/AI_Study/chatbot_app/chatbot_app/db.sqlite3` - sqlite database
- `coursework/AI_Study/chatbot_app/chatbot_app/chatbot/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app/chatbot_app/chatbot/migrations/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app/chatbot_app/config/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app/modules/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app/results/.env.example` - environment secrets
- `coursework/AI_Study/chatbot_app1/chatbot_app/db.sqlite3` - sqlite database
- `coursework/AI_Study/chatbot_app1/chatbot_app/chatbot/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app1/chatbot_app/chatbot/migrations/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app1/chatbot_app/config/__pycache__` - python cache
- `coursework/AI_Study/chatbot_app1/modules/__pycache__` - python cache
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/.env` - environment secrets
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/chatbot_app/staticfiles` - generated static files
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/chatbot_app/db.sqlite3` - sqlite database
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/chatbot_app/chatbot/__pycache__` - python cache
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/chatbot_app/chatbot/migrations/__pycache__` - python cache
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/chatbot_app/config/__pycache__` - python cache
- `coursework/AI_Study/pylib/__pycache__` - python cache
- `coursework/AI_Study/pylib/sample_pac/__pycache__` - python cache
- `coursework/AI_Study/pylib/sample_pac/ab/__pycache__` - python cache
- `coursework/AI_Study/pylib/sample_pac/cd/__pycache__` - python cache
- `coursework/AI_Study/프로젝트 1/.ipynb_checkpoints` - jupyter checkpoint
- `coursework/AI_Study/프로젝트 1/catboost_info/learn/events.out.tfevents` - training event log
- `coursework/AI_Study/프로젝트 1/catboost_info/test/events.out.tfevents` - training event log
- `coursework/AI_Study/프로젝트 1/model/lgb_model.pkl` - model/data artifact
- `coursework/AI_Study/프로젝트 2/5. Module/__pycache__` - python cache
- `experiments/ai-projects/.ipynb_checkpoints` - jupyter checkpoint
- `experiments/short-code/.ipynb_checkpoints` - jupyter checkpoint
- `projects/country-economy-project/.ipynb_checkpoints` - jupyter checkpoint
- `projects/country-economy-project/data/.ipynb_checkpoints` - jupyter checkpoint
- `projects/economy-project/.ipynb_checkpoints` - jupyter checkpoint
- `projects/world-happiness/.ipynb_checkpoints` - jupyter checkpoint

## 4. 민감정보 후보

주의: 실제 키/비밀번호 값은 출력하지 않음.

- `CLEANUP_REPORT.md` - content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `README.md` - content keywords: PASSWORD, TOKEN; 값은 출력하지 않음
- `requirements.txt` - content keywords: TOKEN; 값은 출력하지 않음
- `archive/cleanup-candidates/README.md.md` - content keywords: HUGGINGFACE; 값은 출력하지 않음
- `coursework/AI_Study/01_python/ch06_모듈과 패키지.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/01_python/ch14_웹데이터수집1_정적_공공api.ipynb` - path keywords: api; 값은 출력하지 않음
- `coursework/AI_Study/01_python/ch15_데이터베이스연동.ipynb` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/02_HTML_CSS/ch03_HTML-2_form공간분할/1_form태그기본.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/02_HTML_CSS/ch03_HTML-2_form공간분할/2_입력양식type들.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/02_HTML_CSS/ch03_HTML-2_form공간분할/4_pw유효성검사.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/02_HTML_CSS/ch03_HTML-2_form공간분할/quiz.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/02_HTML_CSS/ch04_CSS선택자/5_속성선택자.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/03_JavaScript/aJax/3_keyup.html` - path keywords: key; content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/03_JavaScript/aJax/join.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/03_JavaScript/ch08_기본객체/quiz_회원가입.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/03_JavaScript/ch10_문서객체/2_문서객체가져오기3_name.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/03_JavaScript/ch11_이벤트/7_이벤트제한(표준이벤트모델).html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/03_JavaScript/ch11_이벤트/7_이벤트제한.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/07_자연어처리/ch1_NLTK자연어처리.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/07_자연어처리/ch2_한글형태소분석_시각화.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/07_자연어처리/ch3_한글형태소분석_시각화_유사도분석_Quiz.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/07_자연어처리/ch4_RNN(RecurrentNeuralNetwork;순환신경망).ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/3_summaryWeb.py` - content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/4_chatbotWeb.py` - content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/6_assistantWebbot.py` - content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ai_llm.py` - content keywords: API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch 9 크로마를 이용한.ipynb` - content keywords: API_KEY, PINECONE_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch1 허깅페이스.ipynb` - content keywords: API_KEY, HF_TOKEN, HUGGINGFACE, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch1.허깅페이스.ipynb` - content keywords: API_KEY, HF_TOKEN, HUGGINGFACE, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch2.Ollama_LLM활용의기본개념(LangChain).ipynb` - content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch3_OpenAI Chat Completions API.ipynb` - path keywords: api; content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch4_OpenAI_Dall-e_API.ipynb` - path keywords: api; content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch5_OpenAI_TTS_API.ipynb` - path keywords: api; content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch6_OpenAI_Whisper_API.ipynb` - path keywords: api; content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch7_OpenAI_Assistants_API.ipynb` - path keywords: api; content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch8_function_calling.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.01_vectorEmbeddingModel성능비교.ipynb` - content keywords: API_KEY, OPENAI_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.02_ChatOpenAI와 렝체인을 활용한 검증.ipynb` - content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.03_ChatUpstage와 렝체인을 활용한 검증.ipynb` - content keywords: API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.05_vectorDatabase저장없이 RAG구현.ipynb` - content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.06_LangChain과 vectorDatabase을 활용한 RAG구현.ipynb` - content keywords: API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.07_LangChain과 vectorDatabase을 활용한 RAG구현(UpstageEmbedding).ipynb` - content keywords: API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.08_chroma→pinecone을 활용한 RAG구현.ipynb` - content keywords: API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.09_chroma→pinecone을 활용한 RAG구현(upstage Embedding).ipynb` - content keywords: API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.10_Retrieval의 효율개선을 위한 전처리(table).ipynb` - content keywords: API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.11_Retrieval의 효율개선을 위한 전처리(markdown)_키워드사전활용.ipynb` - content keywords: API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.12_Retrieval의 성능개선을 위한 metadata활용.ipynb` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/08_LLM/ch9.13_Retrieval의 성능개선을 위한 metadata활용(rerank1방법).ipynb` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/09_MachineLearning/1장_머신러닝시작하기.ipynb` - content keywords: API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch1/data/apt_api.csv` - path keywords: api; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_CRUD/filters.py` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_CRUD/templates/1_onlyget/join.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_methods_error/app.py` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_methods_error/ex1_get_error.py` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_methods_error/filters.py` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_methods_error/templates/1_onlyget/join.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_methods_error/templates/2_crud/join.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch3_methods_error/templates/2_crud/result.html` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch5_dbtest/database/test_mysql.ipynb` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch5_dbtest/database/test_oracle.ipynb` - content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch6_todo/app.py` - content keywords: SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/11_flask/ch6_todo/database/connection.py` - path keywords: connection.py; content keywords: PASSWORD; 값은 출력하지 않음
- `coursework/AI_Study/12_django/ch01_hello/ch01/settings.py` - path keywords: settings.py; content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/12_django/ch02_wordcnt/ch02/settings.py` - path keywords: settings.py; content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/article/templates/article/article_confirm_delete.html` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/article/templates/article/article_form.html` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/book/templates/book/book_confirm_delete.html` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/book/templates/book/book_form.html` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/filetest/templates/filetest/fileupload.html` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/myproject/settings.py` - path keywords: settings.py; content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/12_django/myproject/myproject/templates/header.html` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app/requirements.txt` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app/chatbot_app/rag_module.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app/chatbot_app/web_chatbot.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app/chatbot_app/config/settings.py` - path keywords: settings.py; content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app/modules/rag_module.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app/results/.env.example` - path keywords: .env; content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/12. 프롬프트 엔지니어링.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/RAGAS_TEST_refactored_patched.ipynb` - content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/RAGAS_TEST_refactored_patched_faithfulness_only.ipynb` - content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/RAGAS_TEST_refactored_patched_orthodox_callablefix2.ipynb` - content keywords: API_KEY, OPENAI_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/RAGAS_TEST_refactored_prompt_compare_patched.ipynb` - content keywords: API_KEY, OPENAI_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/chatbot_app/config/settings.py` - path keywords: settings.py; content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/chatbot_app1/modules/rag_module.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/.env` - path keywords: .env; content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, SECRET_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/README.md` - content keywords: API_KEY, HF_TOKEN, HUGGINGFACE, OPENAI_API_KEY, PINECONE_API_KEY, SECRET_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/requirements.txt` - content keywords: HUGGINGFACE, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/chatbot_app/config/settings.py` - path keywords: settings.py; content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `coursework/AI_Study/lease_law_app-main/lease_law_app-main/modules/rag_module.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag_module.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/Untitled.ipynb` - content keywords: API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/final comparison/final_module_cld.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/final comparison/final_module_gmn.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/final comparison/final_module_gpt.py` - content keywords: API_KEY, OPENAI_API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag optimizing/rag_llm_pipeline.py` - content keywords: API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag optimizing/rag_module.py` - content keywords: API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag optimizing/rag_module_cl.py` - content keywords: API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag optimizing/rag_module_cl2.py` - content keywords: API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag optimizing/rag_module_ge.py` - content keywords: API_KEY, PINECONE_API_KEY, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/rag optimizing/rag_module_ge2.py` - content keywords: API_KEY, PINECONE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/solar+bm25/improved_module_cg.py` - content keywords: API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/solar+bm25/improved_module_cl.py` - content keywords: API_KEY, PINECONE_API_KEY, TOKEN; 값은 출력하지 않음
- `coursework/AI_Study/프로젝트 2/5. Module/solar+bm25/improved_module_ge.py` - content keywords: API_KEY, PINECONE_API_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `docs/project-index.md` - content keywords: PASSWORD, SECRET_KEY; 값은 출력하지 않음
- `experiments/short-code/python_tetris_jupyter.ipynb` - content keywords: TOKEN; 값은 출력하지 않음
- `experiments/vibe-project/password_tool_jupyter-checkpoint.ipynb` - path keywords: password; content keywords: PASSWORD; 값은 출력하지 않음
- `experiments/vibe-project/password_tool_jupyter.ipynb` - path keywords: password; content keywords: PASSWORD; 값은 출력하지 않음
- `experiments/vibe-project/Secret_Key-checkpoint.ipynb` - path keywords: key, secret; 값은 출력하지 않음
- `experiments/vibe-project/Secret_Key.ipynb` - path keywords: key, secret; 값은 출력하지 않음
- `programming/README.md` - content keywords: PASSWORD, TOKEN; 값은 출력하지 않음
- `programming/ROADMAP.md` - content keywords: PASSWORD; 값은 출력하지 않음
- `programming/06_backend/rest_api_practice/README.md` - path keywords: api; 값은 출력하지 않음
- `programming/09_security_basics/README.md` - content keywords: PASSWORD; 값은 출력하지 않음
- `reports/repo_cleaner_report.json` - content keywords: API_KEY, DATABASE_URL, HF_TOKEN, HUGGINGFACE, OPENAI_API_KEY, PASSWORD, PINECONE_API_KEY, SECRET_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `reports/repo_cleaner_report.md` - content keywords: API_KEY, DATABASE_URL, HF_TOKEN, HUGGINGFACE, OPENAI_API_KEY, PASSWORD, PINECONE_API_KEY, SECRET_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `tools/repo_cleaner/repo_cleaner.py` - content keywords: API_KEY, DATABASE_URL, HF_TOKEN, HUGGINGFACE, OPENAI_API_KEY, PASSWORD, PINECONE_API_KEY, SECRET_KEY, TOKEN, UPSTAGE_API_KEY; 값은 출력하지 않음
- `tools/repo_cleaner/sample_output.md` - content keywords: SECRET_KEY; 값은 출력하지 않음

## 5. 대용량 파일 후보

| 경로 | 크기 | 확장자 | 추천 조치 |
|---|---:|---|---|
| `coursework/AI_Study/프로젝트 1/data/전처리/1_1_법정동_일별_날씨_전력량.csv` | 43.66 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |
| `coursework/AI_Study/프로젝트 1/data/전처리/3_일부_법정동제외_이상치보간_일별_전력량.csv` | 42.70 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |
| `coursework/AI_Study/프로젝트 1/data/전처리/2_일부_법정동제외_일별_전력량.csv` | 42.69 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |
| `coursework/AI_Study/02_HTML_CSS/ch02_HTML-1_글자목록표img/ch02_sound/sonatina.mp4` | 18.43 MB | .mp4 | Git LFS 또는 외부 보관 |
| `coursework/AI_Study/프로젝트 1/data/훈련데이터셋/3_훈련데이터셋_LAG_휴일포함.csv` | 16.04 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |
| `coursework/AI_Study/프로젝트 1/data/훈련데이터셋/2_훈련데이터셋_LAG포함.csv` | 14.78 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |
| `coursework/AI_Study/chatbot_app/kor.traineddata` | 14.61 MB | .traineddata | Git LFS 또는 외부 보관 |
| `coursework/AI_Study/chatbot_app1/kor.traineddata` | 14.61 MB | .traineddata | Git LFS 또는 외부 보관 |
| `coursework/AI_Study/01_python/ch13_예2_사례연구_서울부산 상가분석.ipynb` | 12.69 MB | .ipynb | 수동 확인 |
| `coursework/AI_Study/01_python/ch13_예1_포트폴리오_아파트분양가분석.ipynb` | 12.10 MB | .ipynb | 수동 확인 |
| `coursework/AI_Study/프로젝트 1/data/전처리/1_법정동_일별_전력량.csv` | 11.38 MB | .csv | Git LFS / 외부 보관 / 수동 확인 |

## 6. 정리 우선순위 추천

1. 민감정보 후보(.env, Secret/API/Token/Password 키워드 파일)를 먼저 수동 검토
2. venv/.venv 폴더는 재현 가능한 requirements로 대체 가능한지 검토
3. __pycache__ 및 .pyc 추적 제외/정리 검토
4. Chroma DB는 재생성 가능 여부 확인 후 Git 제외 또는 외부 보관 검토
5. 10MB 이상 데이터/모델/미디어 파일은 Git LFS 또는 외부 보관 기준 결정
6. 용량이 큰 최상위 폴더부터 분리 기준 수립: coursework, projects, reports

## 7. 다음 작업 추천

- 민감정보 후보를 사람이 직접 열어 실제 값 포함 여부를 확인합니다.
- `.gitignore` 규칙을 정리하기 전에 이미 추적 중인 파일 목록을 별도로 확인합니다.
- 대용량 데이터/모델은 Git LFS, 외부 보관, 재생성 스크립트 중 하나로 관리 기준을 정합니다.
- README 초안은 JSON 보고서의 `readme_summary` 항목을 바탕으로 작성합니다.
