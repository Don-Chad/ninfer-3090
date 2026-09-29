#include <httplib.h>

#include <iostream>
#include <string>
#include <thread>

int main() {
    httplib::Server server;
    server.set_payload_max_length(16384);
    server.Post("/payload", [](const httplib::Request& request, httplib::Response& response) {
        response.set_content(std::to_string(request.body.size()), "text/plain");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        std::cerr << "could not bind HTTP test server\n";
        return 1;
    }
    std::thread worker([&] { server.listen_after_bind(); });
    server.wait_until_ready();
    httplib::Client client("127.0.0.1", port);
    const auto below = client.Post("/payload", std::string(9000, 'a'),
                                   "application/x-www-form-urlencoded");
    const auto above = client.Post("/payload", std::string(17000, 'a'),
                                   "application/x-www-form-urlencoded");
    const auto json = client.Post("/payload", std::string(9000, 'a'), "application/json");
    server.stop();
    worker.join();
    if (!below || below->status != 200 || below->body != "9000" || !above ||
        above->status != 413 || !json || json->status != 200 || json->body != "9000") {
        std::cerr << "form below cap=" << (below ? below->status : -1)
                  << ", form above cap=" << (above ? above->status : -1)
                  << ", json below cap=" << (json ? json->status : -1) << '\n';
        return 1;
    }
    std::cout << "ok\n";
}
