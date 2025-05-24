from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Event
import asyncio


class ProcessingEvent(Event):
    intermediate_result: str

class MyWorkflow(Workflow):
    @step
    async def stepB(self, ev: ProcessingEvent) -> StopEvent:
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result= final_result)

    @step
    async def stepA(self, ev:StartEvent) -> ProcessingEvent:
        return ProcessingEvent(intermediate_result= " step A complete")
    
   

w = MyWorkflow(timeout= 10, verbose = True)
async def main ():
    result = await w.run()
    print(result)

asyncio.run(main())