type LayoutProps = {
    children: React.ReactNode;
}

export const Layout = ({ children }: LayoutProps) => {
    return (
        <div className="min-h-screen bg-[#1A0A0A] flex flex-col items-center">
            {children}
        </div>
    );
};
