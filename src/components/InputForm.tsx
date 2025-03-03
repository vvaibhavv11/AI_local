import React, { useState } from "react";
import { SendHorizonalIcon, Paperclip } from "lucide-react";

type InputFormProps = {
    onSubmit: (content: string) => void;
}

export const InputForm = ({ onSubmit }: InputFormProps) => {
    const [inputValue, setInputValue] = useState<string>("");
    const [attachment, setAttachment] = useState<File | null>(null);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!inputValue.trim()) return;
        onSubmit(inputValue.trim());
        setInputValue("");
        setAttachment(null);
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setAttachment(file);
        }
    };

    return (
        <form onSubmit={handleSubmit} className="relative flex-1 max-w-2xl">
            <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask me anything..."
                className="w-full bg-[#2A1515] text-[#D4A6A6] rounded-full py-4 pl-6 pr-24 focus:outline-none focus:ring-2 focus:ring-[#D4A6A6] focus:ring-opacity-50 placeholder:text-[#D4A6A6] placeholder:opacity-50"
            />
            <div className="absolute right-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
                <label className="cursor-pointer text-[#D4A6A6] hover:text-white">
                    <input
                        type="file"
                        onChange={handleFileChange}
                        className="hidden"
                        accept="image/*,.pdf,.doc,.docx,.txt"
                    />
                    <Paperclip className="w-5 h-5" />
                </label>
                <button type="submit" className="text-[#D4A6A6] hover:text-white">
                    <SendHorizonalIcon className="w-6 h-6" />
                </button>
            </div>
            {attachment && (
                <div className="absolute left-6 -top-8 text-sm text-[#D4A6A6]">
                    Attached: {attachment.name}
                </div>
            )}
        </form>
    );
};
