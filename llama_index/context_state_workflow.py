from llama_index.core.workflow import Context, StartEvent, StopEvent
from llama_index.core.workflow import Workflow, step, Event
import asyncio


class MyWorkflow(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
        await ctx.set("query", "What is the capital of France?")

        for i in range(3):
            await ctx.set("query", f"tentative {i}")
        # retrieve query from the context
        query = await ctx.get("query")
        print(query)
        return StopEvent(result=query )
    
w = MyWorkflow(timeout= 10)



async def main():
    result = await w.run()


asyncio.run(main())