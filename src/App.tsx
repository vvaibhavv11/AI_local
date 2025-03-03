import { scan } from "react-scan"; // must be imported before React and React DOM
import { useState } from "react";
import { open } from '@tauri-apps/plugin-dialog';
import { invoke, Channel } from '@tauri-apps/api/core';
import { Layout } from './components/Layout';
import { Header } from './components/Header';
import { Button } from './components/Button';
import { InputForm } from './components/InputForm';
import { Footer } from './components/Footer';
import { Toast } from './components/Toast';
import { MessageList } from './components/messages/MessageList';
import { aiResponse, Message, MessageType } from './types';
import { v4 as uuidv4 } from 'uuid';
scan({
  enabled: false,
});

function App() {
    const [modelLoaded, setModelLoaded] = useState<boolean>(false);
    const [showToast, setShowToast] = useState<boolean>(false);
    const [messages, setMessages] = useState<Message[]>([]);

    const openFileSelector = async () => {
        const modelPath = await open({
            canCreateDirectories: false,
            title: "Select model file gguf",
            multiple: false,
            directory: false,
            filters: [{
                extensions: ['gguf'],
                name: "gguf"
            }]
        });

        if (modelPath === null) {
            console.log("not selected");
            return;
        }

        const isLoaded = await invoke('load_model', { model_path: modelPath });
        console.log(isLoaded);
        if (isLoaded) {
            setModelLoaded(true);
            setShowToast(true);
            setTimeout(() => setShowToast(false), 3000);
        }
    };

    const handleEject = () => {
        setModelLoaded(false);
    };

    const formatMessagesForContext = (messages: Message[]): string => {
        return messages.map(msg =>
            `<|im_start|>${msg.type === 'user' ? 'user' : 'assistant'}\n${msg.content}<|im_end|>\n`
        ).join('');
    };

    const handleSubmit = async (content: string) => {
        const userMessage: Message = {
            id: uuidv4(),
            content,
            type: "user",
            isLoading: false,
        };

        setMessages(prev => [...prev, userMessage]);

        // Create initial AI message with loading state
        const aiMessageId = uuidv4();
        const aiMessage: Message = {
            id: aiMessageId,
            content: '▋', // Using a block character as typing indicator
            type: MessageType.AI,
            isLoading: true // Add this to Message type
        };

        setMessages(prev => [...prev, aiMessage]);

        // Format previous context with the new format
        const conversationHistory = formatMessagesForContext(messages);
        const fullPrompt = `${conversationHistory}<|im_start|>user\n${content}<|im_end|>\n<|im_start|>assistant\n`;

        const onEvent = new Channel<aiResponse>();
        let fullResponse = '';

        onEvent.onmessage = (res) => {
            fullResponse += res.data.response;
            setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                    ? { ...msg, content: fullResponse, isLoading: false }
                    : msg
            ));
        };

        try {
            await invoke('response', { input: fullPrompt, on_event: onEvent });
        } catch (error) {
            console.error('Error getting AI response:', error);
            setMessages(prev => prev.map(msg =>
                msg.id === aiMessageId
                    ? { ...msg, content: 'Error generating response. Please try again.', isLoading: false }
                    : msg
            ));
        }
    };

    return (
        <Layout>
            <Toast show={showToast} message="Model loaded successfully" />
            <Header />

            <div className="flex-1 w-full max-w-4xl px-4">
                <MessageList messages={messages} />
                {/* Messages will appear here */}
            </div>

            <div className="w-full max-w-4xl flex justify-center items-center space-x-4 px-4 mb-8">
                <Button
                    modelLoaded={modelLoaded}
                    onClick={modelLoaded ? handleEject : openFileSelector}
                />
                <InputForm onSubmit={handleSubmit} />
            </div>

            <Footer />
        </Layout>
    );
}

export default App;
