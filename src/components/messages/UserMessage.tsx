type UserMessageProps = {
    content: string;
}

export const UserMessage = ({ content }: UserMessageProps) => {
    return (
        <div className="flex justify-end mb-4">
            <div className="bg-[#2A1515] text-[#D4A6A6] rounded-2xl rounded-tr-none px-6 py-4 max-w-[80%]">
                <p className="text-sm">{content}</p>
            </div>
        </div>
    );
};

