
import Markdown from "react-markdown";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { useEffect, useState } from "react";
import { Components } from "react-markdown/lib/ast-to-react";

type AIMessageProps = {
    content: string;
    isLoading?: boolean;
}

export const AIMessage = ({ content, isLoading }: AIMessageProps) => {
    const [loadingDots, setLoadingDots] = useState('▋');

    useEffect(() => {
        if (isLoading) {
            const interval = setInterval(() => {
                setLoadingDots(prev => prev === '▋' ? '▋▋' : '▋');
            }, 500);
            return () => clearInterval(interval);
        }
    }, [isLoading]);

    const components: Components = {
        code({ className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            const code = String(children).replace(/\n$/, '');

            if (match) {
                return (
                    <div className="relative group my-4">
                        <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                                onClick={() => navigator.clipboard.writeText(code)}
                                className="bg-[#4A3535] hover:bg-[#5A4545] text-[#D4A6A6]
                                         rounded px-2 py-1 text-xs transition-colors"
                            >
                                Copy
                            </button>
                        </div>
                        <SyntaxHighlighter
                            style={vscDarkPlus}
                            language={match[1]}
                            PreTag="div"
                            className="rounded-lg !bg-[#2A1515] !mt-0 !mb-0"
                            customStyle={{
                                margin: 0,
                                padding: '1rem',
                                backgroundColor: '#2A1515'
                            }}
                        >
                            {code}
                        </SyntaxHighlighter>
                    </div>
                );
            }
            return (
                <code
                    className="bg-[#2A1515] px-1.5 py-0.5 rounded font-mono text-[#D4A6A6]"
                    {...props}
                >
                    {children}
                </code>
            );
        },
        p({ children }) {
            return <p className="mb-4 last:mb-0">{children}</p>
        },
        ul({ children }) {
            return <ul className="list-disc ml-4 mb-4">{children}</ul>
        },
        ol({ children }) {
            return <ol className="list-decimal ml-4 mb-4">{children}</ol>
        },
        li({ children }) {
            return <li className="mb-1">{children}</li>
        },
        h1({ children }) {
            return <h1 className="text-xl font-bold mb-4">{children}</h1>
        },
        h2({ children }) {
            return <h2 className="text-lg font-bold mb-3">{children}</h2>
        },
        h3({ children }) {
            return <h3 className="text-base font-bold mb-2">{children}</h3>
        }
    };

    return (
        <div className="flex justify-start mb-4">
            <div className="bg-[#3A2525] text-[#D4A6A6] rounded-2xl rounded-tl-none px-6 py-4 max-w-[80%]">
                <div className="text-sm">
                    {isLoading ? (
                        <span className="animate-pulse">{loadingDots}</span>
                    ) : (
                        <Markdown components={components}>
                            {content}
                        </Markdown>
                    )}
                </div>
            </div>
        </div>
    );
};
