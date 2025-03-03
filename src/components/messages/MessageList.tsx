import { Message, MessageType } from '../../types/index';
import { UserMessage } from './UserMessage';
import { AIMessage } from './AIMessage';

type MessageListProps = {
    messages: Message[];
}

export const MessageList = ({ messages }: MessageListProps) => {
    return (
        <div className="flex flex-col space-y-4 overflow-y-auto">
            {messages.map((message) => (
                message.type === MessageType.AI
                    ? <AIMessage key={message.id} content={message.content} isLoading={message.isLoading} />
                    : <UserMessage key={message.id} content={message.content} />
            ))}
        </div>
    );
};
