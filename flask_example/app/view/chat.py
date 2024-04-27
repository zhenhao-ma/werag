from flask import (
    Blueprint, render_template, request, current_app, flash
)
from flask import g
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.utils import login_required

bp = Blueprint('chat', __name__, url_prefix='/chat')


@bp.route('/chat', methods=('GET', 'POST'))
@login_required
def chat():
    if request.method == "POST":
        from app.llm import get_llm
        llm = get_llm()

        question = request.form["question"]

        prompt_template = """
        ### [INST] 
        Instruction: 回复下述问题，这里是一些数据和资料供你参考：
        
        {context}
        
        ### QUESTION:
        {question} 
        
        [/INST]
        """

        # Abstraction of Prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Creating an LLM Chain

        llm_chain = LLMChain(llm=llm, prompt=prompt)

        # RAG Chain
        rag_chain = (
                {"context": current_app.config['rag'].as_retriever(user=str(g.user['id']),
                                                                   content_type="user_contents"),
                 "question": RunnablePassthrough()}
                | llm_chain
        )
        answer = rag_chain.invoke(question)['text']
        flash("LLM response")
        return render_template("chat/chat.html", question=question, answer=answer)

    return render_template("chat/chat.html", question="", answer="")
