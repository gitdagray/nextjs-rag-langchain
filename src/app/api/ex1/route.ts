import {
    StreamingTextResponse,
    createStreamDataTransformer
} from 'ai';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { HttpResponseOutputParser } from 'langchain/output_parsers';

export const dynamic = 'force-dynamic'

export async function POST(req: Request) {
    try {
        // Extract the `messages` from the body of the request
        const { messages } = await req.json();
        const message = messages.at(-1).content;

        const prompt = PromptTemplate.fromTemplate("{message}");

        const model = new ChatOpenAI({
            apiKey: process.env.OPENAI_API_KEY!,
            model: 'gpt-3.5-turbo',
            temperature: 0.8,
        });

        /**
       * Chat models stream message chunks rather than bytes, so this
       * output parser handles serialization and encoding.
       */
        const parser = new HttpResponseOutputParser();

        const chain = prompt.pipe(model).pipe(parser);

        // Convert the response into a friendly text-stream
        const stream = await chain.stream({ message });

        //const decoder = new TextDecoder()

        // Each chunk has the same interface as a chat message
        // for await (const chunk of stream) {
        //     //console.log(chunk?.content);
        //     if (chunk) {
        //         console.log(decoder.decode(chunk))
        //     }
        // }

        // Respond with the stream
        return new StreamingTextResponse(
            stream.pipeThrough(createStreamDataTransformer()),
        );
    } catch (e: any) {
        return Response.json({ error: e.message }, { status: e.status ?? 500 });
    }
}