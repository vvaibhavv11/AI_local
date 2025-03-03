export const MessageType = {
    USER: 'user',
    AI: 'ai'
} as const;

export type MessageType = typeof MessageType[keyof typeof MessageType];

export type Message = {
    id: string;
    content: string;
    type: MessageType;
    isLoading: boolean
}

export type aiResponse = {
    event: 'streaming',
    data: {
        response: string
    }
}
