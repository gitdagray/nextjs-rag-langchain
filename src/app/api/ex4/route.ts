import {
    Message as VercelChatMessage,
    StreamingTextResponse,
    createStreamDataTransformer
} from 'ai';
import { ChatOpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import { HttpResponseOutputParser } from 'langchain/output_parsers';

import { JSONLoader } from "langchain/document_loaders/fs/json";
import { RunnableSequence } from '@langchain/core/runnables'
import { formatDocumentsAsString } from 'langchain/util/document';
import { CharacterTextSplitter } from 'langchain/text_splitter';

const loader = new JSONLoader(
    "src/data/states.json",
    ["/state", "/code", "/nickname", "/website", "/admission_date", "/admission_number", "/capital_city", "/capital_url", "/population", "/population_rank", "/constitution_url", "/twitter_url"],
);

export const dynamic = 'force-dynamic'

/**
 * Basic memory formatter that stringifies and passes
 * message history directly into the model.
 */
const formatMessage = (message: VercelChatMessage) => {
    return `${message.role}: ${message.content}`;
};

const TEMPLATE = `Answer the user's questions based only on the following context. If the answer is not in the context, reply politely that you do not have that information available.:
==============================
Context: {context}
==============================
Current conversation: {chat_history}

user: {question}
assistant:`;


export async function POST(req: Request) {
    try {
        // Extract the `messages` from the body of the request
        const { messages } = await req.json();

        const formattedPreviousMessages = messages.slice(0, -1).map(formatMessage);

        const currentMessageContent = messages[messages.length - 1].content;

        const docs = await loader.load();

        // load a JSON object
        // const textSplitter = new CharacterTextSplitter();
        // const docs = await textSplitter.createDocuments([JSON.stringify({
        //     "state": "Kansas",
        //     "slug": "kansas",
        //     "code": "KS",
        //     "nickname": "Sunflower State",
        //     "website": "https://www.kansas.gov",
        //     "admission_date": "1861-01-29",
        //     "admission_number": 34,
        //     "capital_city": "Topeka",
        //     "capital_url": "http://www.topeka.org",
        //     "population": 2893957,
        //     "population_rank": 34,
        //     "constitution_url": "https://kslib.info/405/Kansas-Constitution",
        //     "twitter_url": "http://www.twitter.com/ksgovernment",
        // })]);

        const prompt = PromptTemplate.fromTemplate(TEMPLATE);

        const model = new ChatOpenAI({
            apiKey: process.env.OPENAI_API_KEY!,
            model: 'gpt-3.5-turbo',
            temperature: 0,
            streaming: true,
            verbose: true,
        });

        /**
       * Chat models stream message chunks rather than bytes, so this
       * output parser handles serialization and encoding.
       */
        const parser = new HttpResponseOutputParser();

        const chain = RunnableSequence.from([
            {
                question: (input) => input.question,
                chat_history: (input) => input.chat_history,
                context: () => formatDocumentsAsString(docs),
            },
            prompt,
            model,
            parser,
        ]);

        // Convert the response into a friendly text-stream
        const stream = await chain.stream({
            chat_history: formattedPreviousMessages.join('\n'),
            question: currentMessageContent,
        });

        // Respond with the stream
        return new StreamingTextResponse(
            stream.pipeThrough(createStreamDataTransformer()),
        );
    } catch (e: any) {
        return Response.json({ error: e.message }, { status: e.status ?? 500 });
    }
}