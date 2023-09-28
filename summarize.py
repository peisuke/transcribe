import argparse

import openai
import tqdm
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def correct_text(input_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "], chunk_size=500, chunk_overlap=30
    )
    splitted_original_texts = text_splitter.split_text(input_text)

    formatted_splitted_original_texts = []

    for text in tqdm.tqdm(splitted_original_texts):
        human_template = f"""以下は文の途中で改行が入ってしまっている音声書き起こしの文章です。
読みやすく整理するために、途中で途切れた前後の文を繋げた上で、句読点や改行を追加してください。誤字脱字があれば修正してください。
---
{text}
---
"""

        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct", prompt=human_template, max_tokens=1000
        )

        formatted_splitted_original_texts.append(response.choices[0].text)

    formatted_text = "\n".join([s.strip() for s in formatted_splitted_original_texts])

    return formatted_text


def create_section(input_text):
    list_template = """## 指示文
対象文章に示した内容をセクションに分けたいです。
各セクションのタイトルと、その内容をリスト形式で出力してください。
セクションタイトルは、その内容が類推できるようなものにしてください。
なおフォーマットは以下のようにしてください。
```
- TITLE_#1
- TITLE_#2
- TITLE_#3
- TITLE_#N
```

## 対象文章
{text}"""

    merge_template = """## 指示文
セクションリストに示した内容から、内容が重複すると思われるセクションを削除して、改めてセクションリストとして出力してください。
フォーマットは以下のようにしてください。
```
- TITLE_#1
- TITLE_#2
- TITLE_#3
- TITLE_#N
```

## セクションリスト
{text}"""

    list_prompt = PromptTemplate(template=list_template, input_variables=["text"])
    merge_prompt = PromptTemplate(template=merge_template, input_variables=["text"])

    chat = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
    chain = load_summarize_chain(
        llm=chat,
        chain_type="map_reduce",
        map_prompt=list_prompt,
        combine_prompt=merge_prompt,
        verbose=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=0
    )
    splitted_formatted_texts = text_splitter.split_text(input_text)

    docs = [Document(page_content=t) for t in splitted_formatted_texts]
    ret = chain(inputs=docs, return_only_outputs=True)

    sections = ret["output_text"]
    sections = [s[2:] for s in sections.split("\n") if s.startswith("- ")]

    return sections


def retrive_content(input_text, sections):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "], chunk_size=1000, chunk_overlap=30
    )
    rag_texts = text_splitter.split_text(input_text)

    embeddings = OpenAIEmbeddings()

    docs = [Document(page_content=t) for t in rag_texts]

    db = FAISS.from_documents(docs, embeddings)
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613"),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
    )

    result = []
    for section in tqdm.tqdm(sections):
        query = f"「{section}」について詳しく教えてください"
        resp = qa.run(query)
        result.append({"title": section, "content": resp})
    return result


def predict_title(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=0
    )

    splitted_formatted_texts = text_splitter.split_text(text)

    map_template = """以下の内容を簡潔にまとめてください:
=====
"{text}"
=====
簡潔なまとめ:"""

    merge_template = """## 指示文
以下に示した内容に沿うような、タイトルを日本語で出力してください。

## 内容
{text}"""

    map_prompt = PromptTemplate(template=map_template, input_variables=["text"])
    merge_prompt = PromptTemplate(template=merge_template, input_variables=["text"])

    chat = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
    chain = load_summarize_chain(
        llm=chat,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=merge_prompt,
        verbose=False,
    )

    docs = [Document(page_content=t) for t in splitted_formatted_texts]
    ret = chain(inputs=docs, return_only_outputs=True)["output_text"]

    return ret


def predict_summary(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "], chunk_size=10000, chunk_overlap=0
    )

    splitted_formatted_texts = text_splitter.split_text(text)

    map_template = """以下の内容を簡潔にまとめてください:
=====
"{text}"
=====
簡潔なまとめ:"""

    map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

    chat = ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
    chain = load_summarize_chain(
        llm=chat,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=map_prompt,
        verbose=False,
    )

    docs = [Document(page_content=t) for t in splitted_formatted_texts]
    ret = chain(inputs=docs, return_only_outputs=True)["output_text"]

    return ret


def to_markdown(result):
    # to markdown
    title = result["title"]
    summary = result["summary"]

    md_text = f"# {title}\n\n"

    md_text += "## 概要\n"
    md_text += f"{summary}\n\n"

    for r in result["content"]:
        title = r["title"]
        content = r["content"]
        md_segment = f"## {title}\n{content}\n\n"
        md_text += md_segment

    md_text += "## 全文文字起こし\n"
    md_text += "```\n"
    md_text += result["text"] + "\n"
    md_text += "```"
    return md_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ファイルを処理するスクリプトの説明")
    parser.add_argument("-i", "--input", type=str, required=True, help="処理するファイルの名前")
    parser.add_argument("-o", "--output", type=str, required=True, help="処理するファイルの名前")
    parser.add_argument("-k", "--key", type=str, required=True, help="OpenAI API Key")
    args = parser.parse_args()

    in_filename = args.input
    out_filename = args.output
    apikey = args.key

    with open(in_filename, "r") as f:
        original_text = f.read()

    openai.api_key = apikey

    formatted_text = correct_text(original_text)

    title = predict_title(formatted_text)
    summary = predict_summary(formatted_text)

    sections = create_section(formatted_text)
    contents = retrive_content(formatted_text, sections)

    result = {
        "title": title,
        "summary": summary,
        "text": formatted_text,
        "content": contents,
    }

    markdown = to_markdown(result)
    with open(out_filename, "w") as f:
        f.write(markdown)
