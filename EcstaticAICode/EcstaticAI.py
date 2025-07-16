import os
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from unified_fetcher import UnifiedFinancialFetcher
from retriever import PDFRetriever
from alpha_model import AlphaModel
from backtester import Backtester


class FinanceChatbot:
    def __init__(self, openai_api_key: str):
        os.environ["OPENAI_API_KEY"] = openai_api_key

        print("[Chatbot] Initializing components...")
        self.embedder = OpenAIEmbeddings()
        self.vector_store = FAISS.load_local(
            "faiss_index", self.embedder, allow_dangerous_deserialization=True
        )
        self.retriever = PDFRetriever("faiss_index", self.embedder)
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 3}),
            return_source_documents=True,
        )

        self.fetcher = UnifiedFinancialFetcher(
            yfinance_ticker="AAPL", crypto_symbol="BTC/USD"
        )

    def ask(self, query: str) -> str:
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["price", "stock", "crypto", "macroeconomic", "volatility", "returns", "gdp", "sharpe"]):
            return self._answer_data_question(query)
        elif any(kw in query_lower for kw in ["momentum", "mean reversion", "crossover", "factor"]):
            return self._run_strategy_pipeline(query)
        else:
            return self._answer_knowledge_question(query)

    def _answer_knowledge_question(self, query: str) -> str:
        print("[Chatbot] Routing to PDF knowledge base...\n")
        result = self.qa_chain.invoke({"query": query})
        return f"[PDF Answer]\n{result['result']}"

    def _answer_data_question(self, query: str) -> str:
        print("[Chatbot] Routing to live financial data...\n")
        query_lower = query.lower()

        if "price" in query_lower and "btc" in query_lower:
            return f"BTC/USD latest price: {self.fetcher.get_crypto_price():,.2f}"

        elif "price" in query_lower and "aapl" in query_lower:
            df = self.fetcher.yf.get_price_data()
            close = df["Close"]["AAPL"] if "AAPL" in df["Close"] else df["Close"]
            return f"AAPL latest price: {close.iloc[-1]:,.2f}"

        elif "sharpe" in query_lower:
            stats = self.fetcher.get_stock_summary()
            ratio = stats["sharpe_ratio"]
            if hasattr(ratio, "values"):
                ratio = ratio.values[0]
            elif isinstance(ratio, (list, tuple, np.ndarray)):
                ratio = ratio[0]
            return f"AAPL Sharpe Ratio: {float(ratio):.4f}"

        elif "gdp" in query_lower:
            gdp = self.fetcher.get_macro_data('GDP')
            return f"Latest GDP (FRED/GDP): {gdp.iloc[-1]:,.2f}"

        return "[Data Answer] Sorry, I couldn’t process that financial question."

    def _run_strategy_pipeline(self, query: str) -> str:
        print("[Chatbot] Running alpha strategy and backtest...\n")
        alpha = AlphaModel()

        if "momentum" in query:
            df = alpha.momentum_strategy()
            signal_col = "signal_momentum"
            strategy_name = "Momentum Strategy"

        elif "mean reversion" in query:
            df = alpha.mean_reversion_strategy()
            signal_col = "signal_meanrev"
            strategy_name = "Mean Reversion Strategy"

        elif "crossover" in query:
            df = alpha.moving_average_crossover()
            signal_col = "signal_mac"
            strategy_name = "Moving Average Crossover"

        elif "factor" in query:
            df = alpha.factor_model()
            signal_col = "signal_factor"
            strategy_name = "Factor Model"

        else:
            return "[Alpha] Strategy not recognized. Try: momentum, mean reversion, crossover, or factor."

        bt = Backtester(price_data=df, signal_column=signal_col)
        bt.run()
        metrics = bt.performance_metrics()

        return (
            f"[Strategy Completed] {strategy_name} run and backtested\n"
            f"Total Return: {metrics['Total Return']:.2%}\n"
            f"Max Drawdown: {metrics['Max Drawdown']:.2%}\n"
            f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}"
        )


# Test Block
if __name__ == "__main__":
    print("[Testing FinanceChatbot...]")

    bot = FinanceChatbot(
        openai_api_key="")

    print("\n[Query 1: Finance Theory]")
    print(bot.ask("What is the Black-Scholes formula for option pricing?"))

    print("\n[Query 2: Stock Data]")
    print(bot.ask("What is the latest AAPL stock price?"))

    print("\n[Query 3: Crypto]")
    print(bot.ask("What is the current price of BTC/USD?"))

    print("\n[Query 4: Macro Data]")
    print(bot.ask("Show me the latest GDP number."))

    print("\n[Query 5: Sharpe Ratio]")
    print(bot.ask("What is the Sharpe Ratio for AAPL?"))

    print("\n[Query 6: Alpha Strategy]")
    print(bot.ask("Run momentum strategy and backtest it."))
