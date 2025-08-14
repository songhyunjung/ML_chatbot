import requests
from bs4 import BeautifulSoup

def get_text_from_url(url):
    """
    주어진 URL에서 웹페이지 텍스트를 추출합니다.
    """
    try:
        # 웹페이지에 GET 요청을 보냅니다.
        res = requests.get(url)
        # 응답이 성공적인지 확인합니다.
        res.raise_for_status()
        # BeautifulSoup을 사용하여 HTML을 파싱합니다.
        soup = BeautifulSoup(res.text, 'html.parser')

        # 웹페이지의 모든 텍스트를 추출합니다.
        # 이 예시에서는 제목 태그와 본문 태그를 포함합니다.
        texts = [tag.get_text() for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li'])]
        # 추출된 텍스트들을 하나의 문자열로 합칩니다.
        raw_text = ' '.join(texts)
        # 추출된 텍스트를 반환합니다.
        return raw_text
    except requests.exceptions.RequestException as e:
        # 요청 실패 시 오류 메시지를 출력합니다.
        print(f"Error fetching URL: {e}")
        return None

def preprocess_and_chunk_text(text, chunk_size=500, overlap_size=100):
    """
    텍스트를 전처리하고, 일정 크기의 덩어리로 나눕니다.
    """
    # 텍스트가 None인 경우 빈 리스트를 반환합니다.
    if not text:
        return []

    # 텍스트를 정제합니다. (줄바꿈, 여러 공백 제거 등)
    # 텍스트에서 불필요한 줄바꿈과 공백을 제거하고, 여러 공백을 하나로 만듭니다.
    cleaned_text = ' '.join(text.split())

    chunks = []
    # 텍스트의 길이를 가져옵니다.
    n = len(cleaned_text)
    # 시작 인덱스를 0으로 초기화합니다.
    start = 0
    # 텍스트를 chunk_size만큼 덩어리로 나눕니다.
    while start < n:
        # 현재 덩어리의 끝 인덱스를 계산합니다.
        end = start + chunk_size
        # 텍스트의 끝을 넘지 않도록 end를 조정합니다.
        chunk = cleaned_text[start:end]
        # 덩어리 리스트에 추가합니다.
        chunks.append(chunk)
        # 다음 덩어리를 시작할 위치를 겹치는 부분을 고려하여 설정합니다.
        start += chunk_size - overlap_size
        # 마지막 덩어리가 전체 텍스트보다 짧으면 루프를 종료합니다.
        if start >= n:
            break

    # 덩어리 리스트를 반환합니다.
    return chunks

# 예시 URL
# 실제 머신러닝 관련 문서 URL로 대체해야 합니다.
url = "https://en.wikipedia.org/wiki/Machine_learning"
# 지정된 URL에서 텍스트를 가져옵니다.
data = get_text_from_url(url)

if data:
    # 가져온 텍스트를 전처리하고 청킹합니다.
    chunks = preprocess_and_chunk_text(data)
    # 청크의 개수를 출력합니다.
    print(f"Total chunks created: {len(chunks)}")
    # 첫 번째 청크 내용을 출력합니다.
    print(f"First chunk: {chunks[0]}")
    # 마지막 청크 내용을 출력합니다.
    print(f"Last chunk: {chunks[-1]}")